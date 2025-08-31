import streamlit as st
import sys, os
import json
import spacy
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from collections import Counter
import re
from textstat import flesch_reading_ease
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model and ensure it's available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from NLPP.extractor import fetch_article_text, extract_named_entities

# Load models
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Load question generation pipeline
    try:
        question_generator = pipeline("text2text-generation", 
                                    model="valhalla/t5-small-qg-hl", 
                                    tokenizer="valhalla/t5-small-qg-hl")
    except:
        # Fallback to basic T5 if specialized model fails
        question_generator = None
    
    return t5_model, t5_tokenizer, question_generator, device

class SmartQuestionFilter:
    """Advanced question filtering and validation"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.question_starters = {
            'what', 'who', 'when', 'where', 'why', 'how', 'which', 
            'whose', 'whom', 'did', 'do', 'does', 'is', 'are', 'was', 
            'were', 'will', 'would', 'could', 'should', 'can', 'may'
        }
    
    def is_valid_question_format(self, question):
        """Check if question has valid format"""
        question = question.strip()
        
        # Must end with question mark
        if not question.endswith('?'):
            return False
        
        # Must start with question word or auxiliary verb
        first_word = question.split()[0].lower()
        if first_word not in self.question_starters:
            return False
        
        # Must have reasonable length (between 3 and 25 words)
        word_count = len(question.split())
        if word_count < 3 or word_count > 25:
            return False
        
        return True
    
    def has_sufficient_content(self, question, content):
        """Check if question is answerable from the content"""
        question_words = set(word_tokenize(question.lower()))
        content_words = set(word_tokenize(content.lower()))
        
        # Remove stop words and punctuation
        question_words = {w for w in question_words if w not in self.stop_words and w not in string.punctuation}
        content_words = {w for w in content_words if w not in self.stop_words and w not in string.punctuation}
        
        # Check overlap between question and content
        overlap = len(question_words.intersection(content_words))
        return overlap >= 2  # At least 2 content words should overlap
    
    def is_meaningful_question(self, question, entities):
        """Check if question is meaningful and not too generic"""
        question_lower = question.lower()
        
        # Patterns for overly generic questions
        generic_patterns = [
            r"^what is \w+\?$",
            r"^who is \w+\?$", 
            r"^when did \w+\?$",
            r"^what happened in \w+\?$"
        ]
        
        # Check if question is too generic (single entity without context)
        for pattern in generic_patterns:
            if re.match(pattern, question_lower):
                # Check if the entity in question actually appears meaningfully in content
                entity_in_question = re.findall(r'\b\w+\b', question_lower)[-2]  # Get entity before '?'
                entity_mentions = [ent[0].lower() for ent in entities if entity_in_question in ent[0].lower()]
                if len(entity_mentions) == 0:
                    return False
        
        return True
    
    def remove_duplicate_questions(self, questions):
        """Remove semantically similar questions"""
        unique_questions = []
        seen_patterns = set()
        
        for question in questions:
            # Create a pattern by removing specific entities and keeping structure
            pattern = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 'ENTITY', question)
            pattern = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', 'NUMBER', pattern)
            
            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                unique_questions.append(question)
        
        return unique_questions
    
    def score_question_quality(self, question, content, entities):
        """Score question quality based on multiple factors"""
        score = 0
        question_lower = question.lower()
        
        # Length score (prefer medium-length questions)
        word_count = len(question.split())
        if 5 <= word_count <= 12:
            score += 3
        elif 3 <= word_count <= 15:
            score += 2
        else:
            score += 1
        
        # Question type score
        if question_lower.startswith(('what', 'how')):
            score += 2
        elif question_lower.startswith(('who', 'when', 'where')):
            score += 1.5
        elif question_lower.startswith('why'):
            score += 2.5  # Why questions are often more analytical
        
        # Entity relevance score
        entity_names = [ent[0].lower() for ent in entities]
        for entity in entity_names:
            if entity in question_lower:
                score += 1
        
        # Content relevance score
        if self.has_sufficient_content(question, content):
            score += 2
        
        # Complexity bonus for analytical questions
        analytical_indicators = ['implication', 'effect', 'impact', 'consequence', 'significance', 'analysis']
        if any(indicator in question_lower for indicator in analytical_indicators):
            score += 1.5
        
        return score
    
    def filter_and_rank_questions(self, questions, content, entities, max_questions=20):
        """Main filtering and ranking method"""
        filtered_questions = []
        
        for question in questions:
            if (self.is_valid_question_format(question) and 
                self.is_meaningful_question(question, entities) and
                self.has_sufficient_content(question, content)):
                filtered_questions.append(question)
        
        # Remove duplicates
        filtered_questions = self.remove_duplicate_questions(filtered_questions)
        
        # Score and rank questions
        scored_questions = []
        for question in filtered_questions:
            score = self.score_question_quality(question, content, entities)
            scored_questions.append((score, question))
        
        # Sort by score and return top questions
        scored_questions.sort(reverse=True, key=lambda x: x[0])
        return [question for score, question in scored_questions[:max_questions]]

class AdvancedQuestionGenerator:
    def __init__(self, t5_model, t5_tokenizer, qg_pipeline, device):
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.qg_pipeline = qg_pipeline
        self.device = device
        self.filter = SmartQuestionFilter()
        
    def extract_key_phrases(self, text):
        """Extract key phrases using spaCy's noun chunks and named entities"""
        doc = nlp(text)
        key_phrases = []
        
        # Extract meaningful noun chunks (skip single pronouns, determiners)
        for chunk in doc.noun_chunks:
            if (len(chunk.text.split()) <= 4 and 
                chunk.root.pos_ in ['NOUN', 'PROPN'] and
                len(chunk.text.strip()) > 2):
                key_phrases.append(chunk.text.strip())
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                key_phrases.append(ent.text)
            
        return list(set(key_phrases))
    
    def generate_contextual_wh_questions(self, entities, content):
        """Generate contextually aware WH questions"""
        questions = []
        content_sentences = nltk.sent_tokenize(content)
        
        # Enhanced entity-based questions with context
        entity_context_templates = {
            "PERSON": [
                "What role did {} play in the events described?",
                "How is {} involved in this situation?",
                "What did {} accomplish according to the article?",
                "Why is {} significant in this context?"
            ],
            "ORG": [
                "What is the significance of {} in this article?",
                "How does {} contribute to the main topic?",
                "What actions did {} take according to the report?",
                "What is {}'s position on the matter discussed?"
            ],
            "GPE": [
                "What events occurred in {}?",
                "How does {} relate to the main story?",
                "What is the situation in {} according to the article?",
                "Why is {} mentioned in this context?"
            ],
            "DATE": [
                "What happened on {}?",
                "Why is {} significant in this timeline?",
                "What events are associated with {}?"
            ],
            "EVENT": [
                "What were the consequences of {}?",
                "How did {} affect the situation?",
                "What led to {}?"
            ]
        }
        
        for text, label in entities:
            if label in entity_context_templates:
                # Verify entity appears in meaningful context
                entity_sentences = [s for s in content_sentences if text in s]
                if entity_sentences:  # Only generate if entity has context
                    for template in entity_context_templates[label]:
                        questions.append(template.format(text))
        
        return questions
    
    def generate_inference_questions(self, content):
        """Generate questions that require inference and analysis"""
        questions = []
        
        # Analyze content for different types of information
        doc = nlp(content)
        
        # Look for causal relationships
        causal_indicators = ["because", "due to", "as a result", "caused by", "led to", "resulted in"]
        if any(indicator in content.lower() for indicator in causal_indicators):
            questions.extend([
                "What are the underlying causes mentioned in the article?",
                "What chain of events is described?",
                "How are different events connected in this story?"
            ])
        
        # Look for problems and solutions
        problem_indicators = ["problem", "issue", "challenge", "difficulty", "crisis"]
        solution_indicators = ["solution", "resolve", "address", "tackle", "fix"]
        
        has_problem = any(indicator in content.lower() for indicator in problem_indicators)
        has_solution = any(indicator in content.lower() for indicator in solution_indicators)
        
        if has_problem:
            questions.append("What problems or challenges are identified in the article?")
        if has_solution:
            questions.append("What solutions or approaches are suggested?")
        if has_problem and has_solution:
            questions.append("How effective might the proposed solutions be?")
        
        # Look for future implications
        future_indicators = ["will", "expected", "predict", "forecast", "likely", "potential"]
        if any(indicator in content.lower() for indicator in future_indicators):
            questions.extend([
                "What future developments are anticipated?",
                "What predictions are made in the article?",
                "What are the potential long-term effects?"
            ])
        
        return questions
    
    def generate_critical_thinking_questions(self, content):
        """Generate questions that promote critical thinking"""
        questions = [
            "What evidence is provided to support the main claims?",
            "What perspectives or viewpoints are presented?",
            "What information might be missing from this account?",
            "How reliable are the sources mentioned in the article?",
            "What assumptions underlie the arguments presented?",
            "How might different stakeholders view this situation?",
            "What are the broader implications of these events?",
            "What questions remain unanswered after reading this article?"
        ]
        
        # Add content-specific critical thinking questions
        if "research" in content.lower() or "study" in content.lower():
            questions.extend([
                "What methodology was used in the research described?",
                "What are the limitations of this study?",
                "How generalizable are these findings?"
            ])
        
        if any(word in content.lower() for word in ["policy", "government", "political"]):
            questions.extend([
                "What are the political implications of these events?",
                "How might this affect public policy?",
                "What are the different political perspectives on this issue?"
            ])
        
        return questions
    
    def generate_ml_questions_improved(self, content):
        """Improved ML-based question generation with better prompts"""
        questions = []
        
        # Split content into chunks for better processing
        sentences = nltk.sent_tokenize(content)
        important_sentences = sentences[:3] if len(sentences) >= 3 else sentences
        
        for sentence in important_sentences:
            if len(sentence.split()) > 10:  # Only process substantial sentences
                prompts = [
                    f"generate question: {sentence}",
                    f"create question about: {sentence}",
                    f"what to ask about: {sentence}"
                ]
                
                for prompt in prompts:
                    try:
                        input_ids = self.t5_tokenizer.encode(
                            prompt, 
                            return_tensors="pt", 
                            max_length=512, 
                            truncation=True
                        ).to(self.device)
                        
                        outputs = self.t5_model.generate(
                            input_ids,
                            max_length=50,
                            num_beams=3,
                            early_stopping=True,
                            do_sample=False,  # More deterministic
                            pad_token_id=self.t5_tokenizer.eos_token_id
                        )
                        
                        question = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Clean and validate the generated question
                        if question and len(question.split()) > 3:
                            if not question.endswith('?'):
                                question += '?'
                            questions.append(question)
                            
                    except Exception as e:
                        st.warning(f"Error generating ML question: {e}")
                        continue
        
        return questions
    
    def generate_all_questions(self, entities, content, question_types=None, max_questions=20):
        """Main method to generate all types of questions with smart filtering"""
        if question_types is None:
            question_types = ["contextual_wh", "inference", "critical", "ml"]
        
        all_questions = []
        
        try:
            if "contextual_wh" in question_types:
                all_questions.extend(self.generate_contextual_wh_questions(entities, content))
            
            if "inference" in question_types:
                all_questions.extend(self.generate_inference_questions(content))
            
            if "critical" in question_types:
                all_questions.extend(self.generate_critical_thinking_questions(content))
            
            if "ml" in question_types:
                all_questions.extend(self.generate_ml_questions_improved(content))
            
            # Apply smart filtering and ranking
            final_questions = self.filter.filter_and_rank_questions(
                all_questions, content, entities, max_questions
            )
            
            return final_questions
            
        except Exception as e:
            st.error(f"Error in question generation: {e}")
            return []

def save_questions_to_file(questions, title, format_type="txt"):
    """Save questions to file in different formats"""
    if format_type == "txt":
        content = f"Questions for: {title}\n" + "="*50 + "\n\n"
        content += "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return content
    elif format_type == "json":
        data = {
            "title": title,
            "questions": questions,
            "generated_at": str(pd.Timestamp.now())
        }
        return json.dumps(data, indent=2)

# Streamlit App Interface
st.set_page_config(page_title="Advanced Question Generator", layout="wide")
st.title("🤖 Advanced Automatic Question Generator")
st.markdown("Generate high-quality, contextual questions from news articles using advanced NLP techniques.")

# Load models with error handling
try:
    t5_model, t5_tokenizer, qg_pipeline, device = load_models()
    question_generator = AdvancedQuestionGenerator(t5_model, t5_tokenizer, qg_pipeline, device)
    st.success(f"✅ Models loaded successfully on {device}!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
question_types = st.sidebar.multiselect(
    "Select Question Types:",
    ["contextual_wh", "inference", "critical", "ml"],
    default=["contextual_wh", "inference", "critical"],
    help="Choose which types of questions to generate"
)

max_questions = st.sidebar.slider(
    "Maximum Questions:", 
    min_value=5, 
    max_value=30, 
    value=15,
    help="Maximum number of questions to generate"
)

show_scores = st.sidebar.checkbox(
    "Show Question Quality Scores", 
    value=False,
    help="Display quality scores for generated questions"
)

# Main interface
st.markdown("### 📝 Article Input")
url = st.text_input(
    "🔗 Enter News Article URL", 
    placeholder="https://www.example.com/article",
    help="Paste a news article URL to generate questions from"
)

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🚀 Generate Questions", type="primary"):
        if url.strip():
            with st.spinner("🔄 Processing article and generating questions..."):
                try:
                    # Fetch article
                    article_title, article_content = fetch_article_text(url)
                    
                    if not article_title or not article_content:
                        st.error("❌ Could not extract article content. Please check the URL.")
                        st.stop()
                    
                    # Extract entities
                    entities = extract_named_entities(article_content)
                    
                    # Generate questions
                    questions = question_generator.generate_all_questions(
                        entities, 
                        article_content, 
                        question_types,
                        max_questions
                    )
                    
                    if not questions:
                        st.warning("⚠️ No valid questions could be generated from this article.")
                        st.stop()
                    
                    # Display results
                    st.success(f"✅ Generated {len(questions)} high-quality questions!")
                    
                    # Article info
                    with st.expander("📰 Article Information", expanded=False):
                        st.markdown(f"**Title:** {article_title}")
                        st.markdown(f"**Content Length:** {len(article_content)} characters")
                        st.markdown(f"**Named Entities Found:** {len(entities)}")
                        
                        if entities:
                            entity_display = ", ".join([f"{ent[0]} ({ent[1]})" for ent in entities[:10]])
                            if len(entities) > 10:
                                entity_display += f"... and {len(entities)-10} more"
                            st.markdown(f"**Key Entities:** {entity_display}")
                    
                    # Display questions
                    st.markdown("### 🎯 Generated Questions")
                    
                    if show_scores:
                        # Show questions with quality scores
                        filter_obj = SmartQuestionFilter()
                        for i, question in enumerate(questions, 1):
                            score = filter_obj.score_question_quality(question, article_content, entities)
                            st.markdown(f"{i}. **{question}** _(Quality Score: {score:.1f})_")
                    else:
                        # Show questions without scores
                        for i, question in enumerate(questions, 1):
                            st.markdown(f"{i}. {question}")
                    
                    # Export options
                    questions_text = save_questions_to_file(questions, article_title, "txt")
                    
                    st.markdown("### 📥 Export Options")
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        st.download_button(
                            label="📄 Download as Text",
                            data=questions_text,
                            file_name=f"questions_{article_title[:30].replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    
                    with col_exp2:
                        questions_json = save_questions_to_file(questions, article_title, "json")
                        st.download_button(
                            label="📊 Download as JSON",
                            data=questions_json,
                            file_name=f"questions_{article_title[:30].replace(' ', '_')}.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error processing article: {str(e)}")
                    st.markdown("**Troubleshooting tips:**")
                    st.markdown("- Ensure the URL is accessible and contains readable content")
                    st.markdown("- Try a different news source if the current one is not supported")
                    st.markdown("- Check your internet connection")
        else:
            st.warning("⚠️ Please enter a valid article URL.")

with col2:
    st.markdown("### 💡 Tips")
    st.info("""
    **Best results with:**
    - News articles with clear content
    - Articles with named entities (people, places, organizations)
    - Content with facts and data
    - Articles discussing events or developments
    """)

# Footer information
st.markdown("---")
st.markdown("""
### 🔧 **Advanced Features:**

**Smart Filtering:**
- Validates question format and structure
- Removes incomplete or meaningless questions
- Filters questions based on content relevance
- Eliminates duplicate or similar questions

**Question Types:**
- **Contextual WH**: Who, what, when, where, why, how with context
- **Inference**: Questions requiring logical reasoning
- **Critical Thinking**: Analysis, evaluation, and synthesis questions
- **ML Generated**: AI-generated questions using T5 transformer

**Quality Scoring:**
- Length optimization (5-12 words ideal)
- Question type weighting
- Entity relevance scoring
- Content overlap analysis
- Analytical complexity bonus
""")