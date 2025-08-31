import streamlit as st
import sys, os
import json
import spacy
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from collections import Counter
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

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
# Assuming NER.extractor exists and has the required functions
from NLPP.extractor import fetch_article_text, extract_named_entities

# === Offline Text-to-Speech (pyttsx3) ===
def offline_text_to_speech(text):
    """
    Convert the given text to speech using pyttsx3 and return audio bytes.
    """
    try:
        import pyttsx3
        import tempfile
        import os

        # Create a temporary audio file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()

        # Initialize the TTS engine and synthesize
        engine = pyttsx3.init()
        engine.save_to_file(text, temp_file.name)
        engine.runAndWait()

        # Read and play the audio
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        return audio_bytes

    except Exception as e:
        st.error(f"Error with offline TTS: {e}")
        st.info("Try installing pyttsx3: pip install pyttsx3")
        return None
# === End TTS ===

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

# === EVALUATION METRICS CLASSES ===

class QuestionQualityEvaluator:
    """Comprehensive evaluation system for question quality assessment"""
    
    def __init__(self):
        self.question_types = {
            'factual': ['what', 'who', 'when', 'where', 'which'],
            'analytical': ['why', 'how'],
            'evaluative': ['should', 'would', 'could'],
            'application': ['predict', 'apply', 'use']
        }
        
        self.cognitive_levels = {
            'remember': ['what', 'who', 'when', 'where', 'list', 'name', 'identify'],
            'understand': ['explain', 'describe', 'summarize', 'interpret'],
            'apply': ['apply', 'use', 'demonstrate', 'show'],
            'analyze': ['analyze', 'compare', 'contrast', 'examine', 'why', 'how'],
            'evaluate': ['evaluate', 'assess', 'judge', 'critique', 'justify'],
            'create': ['create', 'design', 'develop', 'generate', 'produce']
        }
    
    def evaluate_question_diversity(self, questions):
        """Evaluate diversity of generated questions"""
        metrics = {}
        
        # Question type distribution
        type_counts = {'factual': 0, 'analytical': 0, 'evaluative': 0, 'application': 0}
        
        for question in questions:
            question_lower = question.lower()
            first_word = question_lower.split()[0] if question_lower else ""
            
            if first_word in self.question_types['factual']:
                type_counts['factual'] += 1
            elif first_word in self.question_types['analytical']:
                type_counts['analytical'] += 1
            elif any(word in question_lower for word in self.question_types['evaluative']):
                type_counts['evaluative'] += 1
            elif any(word in question_lower for word in self.question_types['application']):
                type_counts['application'] += 1
        
        metrics['question_type_distribution'] = type_counts
        metrics['diversity_score'] = 1 - max(type_counts.values()) / len(questions) if questions else 0
        
        return metrics
    
    def evaluate_cognitive_complexity(self, questions):
        """Evaluate cognitive complexity using Bloom's taxonomy"""
        complexity_counts = {level: 0 for level in self.cognitive_levels}
        complexity_scores = []
        
        level_weights = {'remember': 1, 'understand': 2, 'apply': 3, 
                        'analyze': 4, 'evaluate': 5, 'create': 6}
        
        for question in questions:
            question_lower = question.lower()
            question_level = 'remember'  # default
            max_score = 0
            
            for level, keywords in self.cognitive_levels.items():
                if any(keyword in question_lower for keyword in keywords):
                    if level_weights[level] > max_score:
                        max_score = level_weights[level]
                        question_level = level
            
            complexity_counts[question_level] += 1
            complexity_scores.append(max_score)
        
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0
        
        return {
            'cognitive_distribution': complexity_counts,
            'average_complexity': avg_complexity,
            'complexity_scores': complexity_scores
        }
    
    def evaluate_linguistic_quality(self, questions):
        """Evaluate linguistic properties of questions"""
        metrics = {}
        
        word_counts = [len(question.split()) for question in questions]
        readability_scores = []
        
        for question in questions:
            try:
                readability = flesch_reading_ease(question)
                readability_scores.append(readability)
            except:
                readability_scores.append(50)  # neutral score
        
        metrics.update({
            'avg_word_count': np.mean(word_counts) if word_counts else 0,
            'word_count_std': np.std(word_counts) if word_counts else 0,
            'word_count_distribution': word_counts,
            'avg_readability': np.mean(readability_scores) if readability_scores else 0,
            'readability_scores': readability_scores
        })
        
        return metrics
    
    def evaluate_semantic_similarity(self, questions):
        """Evaluate semantic similarity between questions to detect redundancy"""
        if len(questions) < 2:
            return {'avg_similarity': 0, 'similarity_matrix': [], 'redundancy_score': 0}
        
        # Use TF-IDF to compute semantic similarity
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
        try:
            tfidf_matrix = vectorizer.fit_transform(questions)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            avg_similarity = np.mean(upper_triangle)
            
            # Redundancy score: percentage of highly similar pairs (>0.7 similarity)
            redundancy_score = np.sum(upper_triangle > 0.7) / len(upper_triangle) if len(upper_triangle) > 0 else 0
            
            return {
                'avg_similarity': avg_similarity,
                'similarity_matrix': similarity_matrix.tolist(),
                'redundancy_score': redundancy_score,
                'similarity_distribution': upper_triangle.tolist()
            }
        except:
            return {'avg_similarity': 0, 'similarity_matrix': [], 'redundancy_score': 0}
    
    def evaluate_content_relevance(self, questions, article_content):
        """Evaluate how well questions relate to the article content"""
        if not article_content:
            return {'relevance_score': 0, 'keyword_overlap_scores': []}
        
        # Extract keywords from article
        article_words = set(word_tokenize(article_content.lower()))
        article_words = {word for word in article_words if word.isalnum() and len(word) > 3}
        
        relevance_scores = []
        
        for question in questions:
            question_words = set(word_tokenize(question.lower()))
            question_words = {word for word in question_words if word.isalnum() and len(word) > 3}
            
            if len(question_words) == 0:
                relevance_scores.append(0)
            else:
                overlap = len(question_words.intersection(article_words))
                relevance_score = overlap / len(question_words)
                relevance_scores.append(relevance_score)
        
        return {
            'avg_relevance_score': np.mean(relevance_scores) if relevance_scores else 0,
            'keyword_overlap_scores': relevance_scores
        }
    
    def comprehensive_evaluation(self, questions, article_content=""):
        """Perform comprehensive evaluation of question quality"""
        evaluation = {}
        
        # Basic statistics
        evaluation['total_questions'] = len(questions)
        evaluation['timestamp'] = datetime.now().isoformat()
        
        # Diversity evaluation
        diversity_metrics = self.evaluate_question_diversity(questions)
        evaluation['diversity'] = diversity_metrics
        
        # Cognitive complexity evaluation
        complexity_metrics = self.evaluate_cognitive_complexity(questions)
        evaluation['cognitive_complexity'] = complexity_metrics
        
        # Linguistic quality evaluation
        linguistic_metrics = self.evaluate_linguistic_quality(questions)
        evaluation['linguistic_quality'] = linguistic_metrics
        
        # Semantic similarity evaluation
        similarity_metrics = self.evaluate_semantic_similarity(questions)
        evaluation['semantic_similarity'] = similarity_metrics
        
        # Content relevance evaluation
        relevance_metrics = self.evaluate_content_relevance(questions, article_content)
        evaluation['content_relevance'] = relevance_metrics
        
        # Overall quality score (weighted combination)
        overall_score = self.calculate_overall_quality_score(evaluation)
        evaluation['overall_quality_score'] = overall_score
        
        return evaluation

    def calculate_overall_quality_score(self, evaluation):
        """Calculate weighted overall quality score"""
        weights = {
            'diversity': 0.2,
            'complexity': 0.25,
            'linguistic': 0.2,
            'uniqueness': 0.15,  # inverse of redundancy
            'relevance': 0.2
        }
        
        # Normalize scores to 0-1 scale
        diversity_score = evaluation['diversity']['diversity_score']
        complexity_score = evaluation['cognitive_complexity']['average_complexity'] / 6  # max is 6
        linguistic_score = min(evaluation['linguistic_quality']['avg_readability'] / 100, 1)  # cap at 100
        uniqueness_score = 1 - evaluation['semantic_similarity']['redundancy_score']
        relevance_score = evaluation['content_relevance']['avg_relevance_score']
        
        overall_score = (
            weights['diversity'] * diversity_score +
            weights['complexity'] * complexity_score +
            weights['linguistic'] * linguistic_score +
            weights['uniqueness'] * uniqueness_score +
            weights['relevance'] * relevance_score
        )
        
        return overall_score * 100  # Convert to percentage

class VisualizationGenerator:
    """Generate comprehensive visualizations for evaluation metrics"""
    
    def __init__(self, evaluation_data):
        self.data = evaluation_data
    
    def create_overview_dashboard(self):
        """Create overview dashboard with key metrics"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Question Type Distribution', 'Cognitive Complexity Distribution',
                'Word Count Distribution', 'Readability Scores',
                'Similarity Distribution', 'Content Relevance'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Question type distribution (pie chart)
        type_dist = self.data['diversity']['question_type_distribution']
        fig.add_trace(
            go.Pie(labels=list(type_dist.keys()), values=list(type_dist.values()), name="Question Types"),
            row=1, col=1
        )
        
        # Cognitive complexity distribution (bar chart)
        cog_dist = self.data['cognitive_complexity']['cognitive_distribution']
        fig.add_trace(
            go.Bar(x=list(cog_dist.keys()), y=list(cog_dist.values()), name="Cognitive Levels"),
            row=1, col=2
        )
        
        # Word count distribution
        word_counts = self.data['linguistic_quality']['word_count_distribution']
        fig.add_trace(
            go.Histogram(x=word_counts, name="Word Count", nbinsx=10),
            row=1, col=3
        )
        
        # Readability scores
        readability = self.data['linguistic_quality']['readability_scores']
        fig.add_trace(
            go.Histogram(x=readability, name="Readability", nbinsx=10),
            row=2, col=1
        )
        
        # Similarity distribution
        if 'similarity_distribution' in self.data['semantic_similarity']:
            similarity = self.data['semantic_similarity']['similarity_distribution']
            fig.add_trace(
                go.Histogram(x=similarity, name="Similarity", nbinsx=10),
                row=2, col=2
            )
        
        # Content relevance scores
        relevance = self.data['content_relevance']['keyword_overlap_scores']
        fig.add_trace(
            go.Histogram(x=relevance, name="Relevance", nbinsx=10),
            row=2, col=3
        )
        
        fig.update_layout(height=800, title_text="Question Quality Assessment Dashboard")
        return fig
    
    def create_quality_scorecard(self):
        """Create a quality scorecard visualization"""
        metrics = {
            'Diversity Score': self.data['diversity']['diversity_score'] * 100,
            'Avg Complexity': (self.data['cognitive_complexity']['average_complexity'] / 6) * 100,
            'Readability': min(self.data['linguistic_quality']['avg_readability'], 100),
            'Uniqueness': (1 - self.data['semantic_similarity']['redundancy_score']) * 100,
            'Content Relevance': self.data['content_relevance']['avg_relevance_score'] * 100,
            'Overall Quality': self.data['overall_quality_score']
        }
        
        fig = go.Figure()
        
        colors = ['red' if score < 50 else 'orange' if score < 70 else 'green' for score in metrics.values()]
        
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=colors,
            text=[f"{score:.1f}%" for score in metrics.values()],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Quality Scorecard",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=500
        )
        
        return fig
    
    def create_complexity_analysis(self):
        """Create detailed cognitive complexity analysis"""
        complexity_scores = self.data['cognitive_complexity']['complexity_scores']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Complexity Score Distribution', 'Bloom\'s Taxonomy Distribution'],
            specs=[[{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Complexity score distribution
        fig.add_trace(
            go.Histogram(x=complexity_scores, name="Complexity Scores", nbinsx=6),
            row=1, col=1
        )
        
        # Bloom's taxonomy distribution
        cog_dist = self.data['cognitive_complexity']['cognitive_distribution']
        fig.add_trace(
            go.Bar(x=list(cog_dist.keys()), y=list(cog_dist.values()), name="Bloom's Levels"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Cognitive Complexity Analysis")
        return fig

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

# Main navigation tabs
tab1, tab2 = st.tabs(["Question Generator", "Evaluation Metrics"])

# TAB 1: QUESTION GENERATOR
with tab1:
    st.title("🤖 QGEN Question Generator")
    st.markdown("Generate high-quality, contextual questions from news articles using advanced NLP techniques.")

    # Initialize session state for audio data storage
    if 'question_audio_data' not in st.session_state:
        st.session_state.question_audio_data = []

    if 'all_audio_data' not in st.session_state:
        st.session_state.all_audio_data = None

    # Load models with error handling
    try:
        t5_model, t5_tokenizer, qg_pipeline, device = load_models()
        question_generator = AdvancedQuestionGenerator(t5_model, t5_tokenizer, qg_pipeline, device)
        st.success(f"✅ Models loaded successfully on {device}!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

    # Define the fragment for the audio player. This will run independently.
    @st.fragment
    def audio_player_fragment(audio_bytes, index):
        if st.button(f"🔊 Listen to Question {index}", key=f"tts_{index}"):
            st.audio(audio_bytes, format='audio/mp3', start_time=0)

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
                        
                        # Store in session state for evaluation tab
                        st.session_state.article_title = article_title
                        st.session_state.article_content = article_content
                        
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
                            
                        
                        # Store generated questions in session state for evaluation
                        st.session_state.generated_questions = questions
                        st.session_state.entities = entities
                        
                        # Generate and store audio files for all questions if questions exist
                        if questions:
                            with st.spinner("🔊 Generating audio files..."):
                                st.session_state.question_audio_data = []
                                # Create a combined list of questions with their numbers and pauses
                                audio_questions = [f"Question number {i}. {q}." for i, q in enumerate(questions, 1)]
                                all_questions_text = " ".join(audio_questions)
                                
                                for question in questions:
                                    audio = offline_text_to_speech(question)
                                    st.session_state.question_audio_data.append(audio)
                                
                                # Generate and store audio for all questions combined
                                st.session_state.all_audio_data = offline_text_to_speech(all_questions_text)

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
                            
                            # Display questions with TTS buttons
                            st.markdown("### 🎯 Generated Questions")

                            # Add the overall audio player button
                            st.markdown("#### Play all questions")
                            if st.session_state.all_audio_data:
                                st.audio(st.session_state.all_audio_data, format='audio/mp3', start_time=0)
                            else:
                                st.warning("Failed to generate audio for all questions.")
                            
                            st.markdown("---")

                            filter_obj = SmartQuestionFilter() 
                            for i, question in enumerate(questions, 1):
                                with st.container(border=True): 
                                    if show_scores:
                                        # Show questions with quality scores
                                        score = filter_obj.score_question_quality(question, article_content, entities)
                                        st.markdown(f"{i}. **{question}** _(Quality Score: {score:.1f})_")
                                    else:
                                        # Show questions without scores
                                        st.markdown(f"{i}. {question}")
                                    
                                    # Call the fragment here with pre-generated audio
                                    if st.session_state.question_audio_data and st.session_state.question_audio_data[i-1]:
                                        audio_player_fragment(st.session_state.question_audio_data[i-1], i)
                                        
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



    # Footer information
    st.markdown("---")
    

# TAB 2: EVALUATION METRICS
with tab2:
    st.title("📊 Question Quality Evaluation Metrics")
    st.markdown("Comprehensive analysis of generated question quality and effectiveness.")
    
    # Check if we have generated questions from the first tab
    if 'generated_questions' in st.session_state and st.session_state.generated_questions:
        st.info(f"Using {len(st.session_state.generated_questions)} questions from the Question Generator tab.")
        default_questions = st.session_state.generated_questions
        default_content = st.session_state.get('article_content', '')
    else:
        # Provide sample questions for demonstration
        default_questions = [
            "What are the main causes of climate change discussed in the article?",
            "How do renewable energy sources compare to fossil fuels?",
            "Why is international cooperation important for environmental protection?",
            "What role do governments play in environmental policy?",
            "How can individuals contribute to reducing carbon emissions?",
            "What are the economic implications of green technology adoption?",
            "Which countries are leading in renewable energy development?",
            "What challenges face the implementation of clean energy solutions?",
            "How does deforestation impact global climate patterns?",
            "What innovations in sustainable technology are mentioned?"
        ]
        default_content = """
        Climate change represents one of the most pressing challenges of our time, with governments 
        worldwide implementing various renewable energy policies. The transition from fossil fuels 
        to clean energy sources requires substantial international cooperation and economic investment. 
        Deforestation continues to impact global climate patterns, while innovations in sustainable 
        technology offer promising solutions for reducing carbon emissions.
        """
    
    # Evaluation controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Questions to Evaluate")
        questions_text = st.text_area(
            "Enter questions (one per line):", 
            value="\n".join(default_questions),
            height=200,
            help="You can edit these questions or use the ones generated from the Question Generator tab"
        )
    
    with col2:
        st.markdown("### Controls")
        if st.button("🔍 Run Evaluation", type="primary"):
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            if questions:
                # Run evaluation
                evaluator = QuestionQualityEvaluator()
                evaluation_results = evaluator.comprehensive_evaluation(
                    questions, 
                    default_content
                )
                
                # Store results in session state
                st.session_state.evaluation_results = evaluation_results
                st.success(f"✅ Evaluated {len(questions)} questions successfully!")
            else:
                st.warning("Please enter some questions to evaluate.")
    
    # Display results if available
    if 'evaluation_results' in st.session_state:
        results = st.session_state.evaluation_results
        viz_gen = VisualizationGenerator(results)
        
        # Overview metrics
        st.markdown("### 🎯 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Overall Quality Score", 
                f"{results['overall_quality_score']:.1f}%",
                delta=None
            )
        with col2:
            st.metric(
                "Total Questions", 
                results['total_questions']
            )
        with col3:
            st.metric(
                "Avg Complexity", 
                f"{results['cognitive_complexity']['average_complexity']:.1f}/6"
            )
        with col4:
            st.metric(
                "Diversity Score", 
                f"{results['diversity']['diversity_score']:.2f}"
            )
        
        # Quality Scorecard
        st.markdown("### 📈 Quality Scorecard")
        scorecard_fig = viz_gen.create_quality_scorecard()
        st.plotly_chart(scorecard_fig, use_container_width=True)
        
        # Comprehensive Dashboard
        st.markdown("### 📊 Comprehensive Analysis Dashboard")
        dashboard_fig = viz_gen.create_overview_dashboard()
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Detailed Analysis Tabs
        eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
            "🧠 Cognitive Analysis", 
            "🔤 Linguistic Quality", 
            "🎯 Content Relevance", 
            "📋 Detailed Report"
        ])
        
        with eval_tab1:
            complexity_fig = viz_gen.create_complexity_analysis()
            st.plotly_chart(complexity_fig, use_container_width=True)
            
            st.markdown("#### Bloom's Taxonomy Breakdown")
            cog_dist = results['cognitive_complexity']['cognitive_distribution']
            for level, count in cog_dist.items():
                percentage = (count / results['total_questions']) * 100
                st.write(f"**{level.title()}**: {count} questions ({percentage:.1f}%)")
        
        with eval_tab2:
            st.markdown("#### Linguistic Quality Metrics")
            ling_metrics = results['linguistic_quality']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Word Count", f"{ling_metrics['avg_word_count']:.1f}")
                st.metric("Word Count Std Dev", f"{ling_metrics['word_count_std']:.1f}")
            with col2:
                st.metric("Average Readability", f"{ling_metrics['avg_readability']:.1f}")
                
                readability_interpretation = "Easy" if ling_metrics['avg_readability'] > 80 else \
                                           "Moderate" if ling_metrics['avg_readability'] > 60 else "Difficult"
                st.write(f"Readability Level: **{readability_interpretation}**")
        
        with eval_tab3:
            st.markdown("#### Content Relevance Analysis")
            rel_score = results['content_relevance']['avg_relevance_score']
            st.metric("Average Relevance Score", f"{rel_score:.2f}")
            
            relevance_interpretation = "High" if rel_score > 0.6 else \
                                     "Medium" if rel_score > 0.3 else "Low"
            st.write(f"Relevance Level: **{relevance_interpretation}**")
            
            # Individual question relevance scores
            if st.checkbox("Show individual question relevance scores"):
                questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                relevance_scores = results['content_relevance']['keyword_overlap_scores']
                
                relevance_df = pd.DataFrame({
                    'Question': questions[:len(relevance_scores)],
                    'Relevance Score': relevance_scores
                })
                st.dataframe(relevance_df, use_container_width=True)
        
        with eval_tab4:
            st.markdown("#### Detailed Evaluation Report")
            
            # Export evaluation results
            report_data = {
                'evaluation_timestamp': results['timestamp'],
                'total_questions': results['total_questions'],
                'overall_quality_score': results['overall_quality_score'],
                'detailed_metrics': results
            }
            
            st.download_button(
                label="📄 Download Evaluation Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name=f"question_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Summary statistics
            st.markdown("##### Summary Statistics")
            summary_stats = {
                "Question Type Distribution": results['diversity']['question_type_distribution'],
                "Cognitive Complexity": results['cognitive_complexity']['cognitive_distribution'],
                "Semantic Similarity": {
                    "Average Similarity": f"{results['semantic_similarity']['avg_similarity']:.3f}",
                    "Redundancy Score": f"{results['semantic_similarity']['redundancy_score']:.3f}"
                }
            }
            
            for category, stats in summary_stats.items():
                st.markdown(f"**{category}**")
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        st.write(f"  - {key}: {value}")
                st.write("")
    
    else:
        st.info("Generate questions in the Question Generator tab first, or run evaluation on the sample questions provided above.")
        
        # Show explanation of evaluation metrics
        st.markdown("### 📖 About Evaluation Metrics")
        
        with st.expander("Learn about the evaluation criteria", expanded=False):
            st.markdown("""
            **Diversity Score**: Measures how varied the question types are (factual, analytical, evaluative, application)
            
            **Cognitive Complexity**: Based on Bloom's Taxonomy levels:
            - Remember (1): Basic recall questions
            - Understand (2): Comprehension questions  
            - Apply (3): Application-based questions
            - Analyze (4): Analysis and comparison questions
            - Evaluate (5): Evaluation and critique questions
            - Create (6): Creative and synthesis questions
            
            **Linguistic Quality**: 
            - Word count distribution (optimal: 5-12 words)
            - Readability scores using Flesch Reading Ease
            
            **Semantic Similarity**: 
            - Measures redundancy between questions
            - Lower similarity indicates more diverse questions
            
            **Content Relevance**: 
            - Keyword overlap between questions and source content
            - Higher scores indicate better alignment with the article
            """)