# QGEN - Advanced Question Generator

An intelligent question generation tool that creates high-quality, contextual questions from news articles using advanced NLP techniques and machine learning models.

## 🎯 Features

- **Multi-Model Question Generation**: Combines rule-based, template-based, and ML-based approaches
- **Smart Question Filtering**: Advanced filtering system to ensure question quality and relevance
- **Comprehensive Evaluation Metrics**: Built-in quality assessment using multiple criteria
- **Text-to-Speech Integration**: Audio playback for generated questions
- **Interactive Visualizations**: Data-driven insights into question quality
- **Multiple Export Formats**: Save questions as TXT or JSON files
- **Real-time Web Scraping**: Fetch articles directly from URLs

## 🏗️ Architecture

### Core Components

1. **SmartQuestionFilter**: Advanced filtering and validation system
2. **AdvancedQuestionGenerator**: Multi-strategy question generation engine
3. **QuestionQualityEvaluator**: Comprehensive quality assessment framework
4. **VisualizationGenerator**: Interactive data visualization system

### Question Generation Strategies

- **Contextual WH Questions**: Entity-aware questions based on article content
- **Inference Questions**: Questions requiring analytical thinking
- **Critical Thinking Questions**: Questions promoting deeper analysis
- **ML-Generated Questions**: T5 transformer-based question generation

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.8 or higher required
python --version
```

### Install Dependencies

```bash
# Core dependencies
pip install streamlit
pip install torch torchvision torchaudio
pip install transformers
pip install spacy
pip install nltk
pip install scikit-learn
pip install pandas
pip install numpy
pip install plotly
pip install matplotlib
pip install seaborn
pip install textstat

# Web scraping dependencies
pip install selenium
pip install beautifulsoup4
pip install webdriver-manager

# Text-to-speech (optional)
pip install pyttsx3

# Download spaCy model
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

### NLTK Data

The application will automatically download required NLTK data on first run:
- punkt (tokenizer)
- stopwords
- averaged_perceptron_tagger

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

### Basic Workflow

1. **Input Article URL**: Paste a news article URL in the input field
2. **Configure Settings**: Select question types and maximum number of questions
3. **Generate Questions**: Click "Generate Questions" to process the article
4. **Review Results**: Examine generated questions with optional quality scores
5. **Audio Playback**: Listen to individual questions or all questions at once
6. **Export Results**: Download questions as TXT or JSON files
7. **Evaluate Quality**: Use the Evaluation Metrics tab for detailed analysis

### Configuration Options

- **Question Types**:
  - `contextual_wh`: Entity-based WH questions
  - `inference`: Questions requiring inference
  - `critical`: Critical thinking questions
  - `ml`: Machine learning-generated questions

- **Max Questions**: 5-30 questions (default: 15)
- **Show Quality Scores**: Display quality metrics for each question

## 📊 Evaluation Metrics

### Quality Assessment Framework

The system evaluates questions across multiple dimensions:

#### 1. Diversity Metrics
- Question type distribution (factual, analytical, evaluative, application)
- Diversity score (0-1, higher = more diverse)

#### 2. Cognitive Complexity (Bloom's Taxonomy)
- **Remember** (Level 1): Basic recall questions
- **Understand** (Level 2): Comprehension questions
- **Apply** (Level 3): Application-based questions
- **Analyze** (Level 4): Analysis and comparison
- **Evaluate** (Level 5): Evaluation and critique
- **Create** (Level 6): Creative synthesis

#### 3. Linguistic Quality
- Average word count and distribution
- Readability scores (Flesch Reading Ease)
- Question format validation

#### 4. Semantic Similarity
- Redundancy detection using TF-IDF and cosine similarity
- Duplicate question removal

#### 5. Content Relevance
- Keyword overlap between questions and article content
- Entity-based relevance scoring

### Quality Scoring Algorithm

Questions are scored based on:
- Length appropriateness (5-12 words optimal)
- Question type (analytical questions score higher)
- Entity relevance
- Content alignment
- Complexity indicators

## 🏛️ Project Structure

```
qgen/
├── app.py                 # Main Streamlit application
├── NLPP/
│   └── extractor.py      # Article extraction and NER utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── models/              # Cached model files (auto-generated)
```

## 🤖 Models Used

- **T5-Small**: Base text-to-text generation model
- **Valhalla T5-QG**: Specialized question generation model
- **spaCy en_core_web_lg**: Named entity recognition and text processing
- **TF-IDF Vectorizer**: Semantic similarity computation

## 📋 Dependencies

### Core Libraries
- `streamlit`: Web application framework
- `transformers`: Hugging Face transformers for ML models
- `torch`: PyTorch for deep learning
- `spacy`: Advanced NLP processing
- `nltk`: Natural language toolkit
- `scikit-learn`: Machine learning utilities

### Data Processing
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `textstat`: Text readability metrics

### Visualization
- `plotly`: Interactive visualizations
- `matplotlib`: Static plotting
- `seaborn`: Statistical visualizations

### Web Scraping
- `selenium`: Web browser automation
- `beautifulsoup4`: HTML parsing
- `webdriver-manager`: ChromeDriver management

### Optional
- `pyttsx3`: Offline text-to-speech

## ⚙️ Configuration

### Environment Variables
No environment variables required for basic functionality.

### Model Configuration
Models are automatically downloaded and cached on first use:
- T5-Small: ~60MB
- Valhalla T5-QG: ~240MB
- spaCy model: ~50MB

## 🔧 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```

2. **ChromeDriver Issues**
   - Ensure Chrome browser is installed
   - webdriver-manager handles ChromeDriver automatically

3. **NLTK Data Missing**
   - The app automatically downloads required data
   - Manual download: `nltk.download('punkt')`

4. **Memory Issues with Large Articles**
   - Reduce max_questions parameter
   - Consider splitting very long articles

### Performance Optimization

- **GPU Support**: Automatic CUDA detection for faster processing
- **Model Caching**: Models are cached using `@st.cache_resource`
- **Batch Processing**: Efficient processing of multiple sentences

## 📈 Quality Metrics

### Question Quality Indicators

- **High Quality** (Score > 70%):
  - Diverse question types
  - High cognitive complexity
  - Good content relevance
  - Low redundancy

- **Medium Quality** (Score 50-70%):
  - Moderate diversity
  - Some analytical questions
  - Acceptable relevance

- **Low Quality** (Score < 50%):
  - Limited diversity
  - Mostly factual questions
  - High redundancy or low relevance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed description and error logs

## 🔮 Future Enhancements

- Support for multiple article formats (PDF, Word documents)
- Advanced question difficulty calibration
- Multi-language support
- Integration with learning management systems
- Custom question templates
- Collaborative question review features

## 📊 Example Output

```
Generated Questions for: "AI Breakthrough in Healthcare"

1. What role does artificial intelligence play in modern healthcare diagnosis?
2. How do machine learning algorithms improve patient treatment outcomes?
3. Why is data privacy crucial in AI-powered healthcare systems?
4. What are the potential risks of implementing AI in medical decision-making?
5. How might AI transformation affect healthcare accessibility?

Quality Score: 78.5%
- Diversity: High
- Cognitive Complexity: 4.2/6 (Analyze level)
- Content Relevance: 85%
- Redundancy: Low
```

---

**Built with ❤️ for educational technology and intelligent content generation**




con
