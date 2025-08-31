import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

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

# Streamlit Integration
def run_evaluation_interface():
    """Streamlit interface for evaluation metrics"""
    
    st.markdown("## 📊 Question Quality Evaluation Metrics")
    st.markdown("Comprehensive analysis of generated question quality and effectiveness.")
    
    # Sample questions for demonstration (in real use, these would come from the main generator)
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = [
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
    
    if 'article_content' not in st.session_state:
        st.session_state.article_content = """
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
            value="\n".join(st.session_state.generated_questions),
            height=200
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
                    st.session_state.article_content
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 Cognitive Analysis", 
            "🔤 Linguistic Quality", 
            "🎯 Content Relevance", 
            "📋 Detailed Report"
        ])
        
        with tab1:
            complexity_fig = viz_gen.create_complexity_analysis()
            st.plotly_chart(complexity_fig, use_container_width=True)
            
            st.markdown("#### Bloom's Taxonomy Breakdown")
            cog_dist = results['cognitive_complexity']['cognitive_distribution']
            for level, count in cog_dist.items():
                percentage = (count / results['total_questions']) * 100
                st.write(f"**{level.title()}**: {count} questions ({percentage:.1f}%)")
        
        with tab2:
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
        
        with tab3:
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
        
        with tab4:
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

if __name__ == "__main__":
    run_evaluation_interface()