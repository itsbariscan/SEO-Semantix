import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from openai import OpenAI
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Sentence Transformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def read_content_from_file(filename: str) -> Optional[str]:
    """Reads content from the specified file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
        return None

def get_embedding(text: str) -> np.ndarray:
    """Gets embedding for the given text using Sentence Transformers."""
    return st_model.encode([text])[0]

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalizes the embedding vector."""
    return embedding / np.linalg.norm(embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def differential_similarity(content_vector: np.ndarray, term_vector: np.ndarray, neutral_vector: np.ndarray) -> float:
    """Computes differential similarity."""
    return cosine_similarity(content_vector, term_vector) - cosine_similarity(content_vector, neutral_vector)

def get_expanded_semantic_field(keyword: str, perspective: str, n: int = 50, similarity_threshold: float = 0.1) -> List[str]:
    """Generates an expanded semantic field using OpenAI's GPT and filters by similarity."""
    try:
        if perspective == "seo":
            prompt = f"Generate 100 keywords closely related to '{keyword}' from an SEO perspective. Focus on search terms, long-tail keywords, and phrases that users might search for when looking for {keyword}. Provide only the keywords, separated by commas, without numbering or explanations."
        else:
            prompt = f"Generate 100 keywords closely related to '{keyword}'. Focus on types of {keyword}, parts of a {keyword}, materials, styles, and directly associated concepts. Provide only the keywords, separated by commas, without numbering or explanations."

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides closely related keywords."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            temperature=0.5,
        )
        all_terms = response.choices[0].message.content.strip().split(',')
        all_terms = [term.strip() for term in all_terms]
        
        # Compute similarities and filter
        content_vector = normalize_embedding(get_embedding(keyword))
        neutral_vector = normalize_embedding(get_embedding("the"))
        
        term_similarities = []
        for term in all_terms:
            term_vector = normalize_embedding(get_embedding(term))
            similarity = differential_similarity(content_vector, term_vector, neutral_vector)
            term_similarities.append((term, similarity))
        
        # Sort by similarity and filter
        term_similarities.sort(key=lambda x: x[1], reverse=True)
        filtered_terms = [term for term, sim in term_similarities if sim > similarity_threshold]
        
        return filtered_terms[:n]
    except Exception as e:
        logger.error(f"Error getting expanded semantic field: {e}")
        return []

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def visualize_results_interactive(keyword: str, content_vector: np.ndarray, keyword_vector: np.ndarray, 
                                  semantic_field_seo: List[str], semantic_field_semantic: List[str], 
                                  similarities_seo: List[float], similarities_semantic: List[float]) -> go.Figure:
    """Creates two separate bar charts for SEO and General Semantic Fields, ensuring both positive and negative similarities are visible."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("SEO Semantic Field", "General Semantic Field"),
                        vertical_spacing=0.2, row_heights=[0.5, 0.5])
    
    # Sort terms by absolute similarity value
    seo_sorted = sorted(zip(semantic_field_seo, similarities_seo), key=lambda x: abs(x[1]), reverse=True)[:50]
    semantic_sorted = sorted(zip(semantic_field_semantic, similarities_semantic), key=lambda x: abs(x[1]), reverse=True)[:50]
    
    # Debugging: Print out some statistics about the data
    print(f"SEO similarities range: {min(similarities_seo)} to {max(similarities_seo)}")
    print(f"General similarities range: {min(similarities_semantic)} to {max(similarities_semantic)}")
    print(f"Number of negative SEO similarities: {sum(1 for sim in similarities_seo if sim < 0)}")
    print(f"Number of negative General similarities: {sum(1 for sim in similarities_semantic if sim < 0)}")

    # SEO Semantic Field
    positive_seo = [(term, sim) for term, sim in seo_sorted if sim >= 0]
    negative_seo = [(term, sim) for term, sim in seo_sorted if sim < 0]
    
    fig.add_trace(
        go.Bar(x=[term for term, _ in positive_seo], y=[sim for _, sim in positive_seo], 
               name="SEO Positive", marker_color='#1f77b4'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=[term for term, _ in negative_seo], y=[sim for _, sim in negative_seo], 
               name="SEO Negative", marker_color='#d62728'),
        row=1, col=1
    )
    
    # General Semantic Field
    positive_semantic = [(term, sim) for term, sim in semantic_sorted if sim >= 0]
    negative_semantic = [(term, sim) for term, sim in semantic_sorted if sim < 0]
    
    fig.add_trace(
        go.Bar(x=[term for term, _ in positive_semantic], y=[sim for _, sim in positive_semantic], 
               name="General Positive", marker_color='#2ca02c'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=[term for term, _ in negative_semantic], y=[sim for _, sim in negative_semantic], 
               name="General Negative", marker_color='#ff7f0e'),
        row=2, col=1
    )
    
    # Calculate y-axis range to ensure negative values are visible
    y_max = max(max(similarities_seo), max(similarities_semantic))
    y_min = min(min(similarities_seo), min(similarities_semantic))
    y_range = [-max(abs(y_min), y_max), max(abs(y_min), y_max)]

    fig.update_layout(height=1400, title_text=f"Semantic Field Analysis for '{keyword}'")
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(title_text="Differential Similarity", row=1, col=1, range=y_range)
    fig.update_yaxes(title_text="Differential Similarity", row=2, col=1, range=y_range)
    
    # Add a horizontal line at y=0 for both subplots
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0, 
                  xref="paper", yref="y1", line=dict(color="black", width=1))
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0, 
                  xref="paper", yref="y2", line=dict(color="black", width=1))
    
    # Adjust the layout for better readability
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50),
        barmode='relative'  # This will stack positive and negative bars
    )
    
    return fig

def calculate_relevance_score(keyword_vector: np.ndarray, content_vector: np.ndarray) -> float:
    """Calculate a relevance score between the keyword and content."""
    return cosine_similarity(keyword_vector, content_vector)

def generate_comprehensive_analysis(keyword: str, content: str, keyword_vector: np.ndarray, content_vector: np.ndarray, 
                                    semantic_field_seo: List[str], semantic_field_semantic: List[str], 
                                    similarities_seo: List[float], similarities_semantic: List[float]) -> str:
    """Generates a comprehensive analysis with detailed scorecard and actionable SEO recommendations."""
    try:
        relevance_score = calculate_relevance_score(keyword_vector, content_vector)
        
        context = f"""Keyword: {keyword}
Content snippet (first 1000 characters): {content[:1000]}...

Relevance score between keyword and content: {relevance_score:.2f}

SEO Semantic Field (top 30 with similarities):
{', '.join([f"{term} ({sim:.2f})" for term, sim in sorted(zip(semantic_field_seo, similarities_seo), key=lambda x: abs(x[1]), reverse=True)[:30]])}

General Semantic Field (top 30 with similarities):
{', '.join([f"{term} ({sim:.2f})" for term, sim in sorted(zip(semantic_field_semantic, similarities_semantic), key=lambda x: abs(x[1]), reverse=True)[:30]])}

Based on this data, provide a comprehensive analysis that includes:

1. A detailed scorecard evaluating:
   - Keyword-Content Relevance (Score out of 10)
   - SEO Potential (Score out of 10)
   - Content Depth (Score out of 10)
   - Topical Coverage (Score out of 10)
   - Semantic Relevance (Score out of 10)
   - Overall SEO Score (Average of above scores)

   For each aspect, provide a brief explanation of the score and potential areas for improvement.

2. In-depth actionable SEO recommendations:
   - Specific keywords to use more frequently (list at least 5)
   - Suggestions for internal linking (provide at least 3 specific ideas)
   - Content structure improvements (at least 3 suggestions)
   - Topical areas to expand upon (identify at least 3 areas)
   - Potential for featured snippets or rich results (provide specific ideas)
   - Recommendations for meta title and description
   - Suggestions for improving user engagement and dwell time

3. Detailed analysis of the semantic fields:
   - Interpret the positive and negative similarities
   - Identify gaps in the content based on the semantic fields
   - Suggest how to leverage the most relevant terms for content optimization
   - Analyze any unexpected or seemingly unrelated terms and their potential value

4. Competitor analysis suggestions:
   - Key areas to focus on for competitive advantage
   - Potential unique selling points based on the semantic analysis

5. Content strategy recommendations:
   - Long-term content plan based on the semantic fields (at least 3 content ideas)
   - Suggestions for content updates or expansions

Provide your response in HTML format, using appropriate tags for structure and emphasis. Use <h2> for main sections, <h3> for subsections, <ul> and <li> for bullet points, and <strong> for emphasis where appropriate. Use <table> for the scorecard with <th> for headers and <td> for data. Use appropriate classes for styling different sections (e.g., class="scorecard", class="recommendations", etc.).
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert SEO analyst and content strategist with deep knowledge of content optimization, search engine algorithms, and digital marketing strategies. Provide a comprehensive analysis with actionable recommendations based on semantic field analysis and content relevance."},
                {"role": "user", "content": context}
            ],
            max_tokens=2500,
            n=1,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating comprehensive analysis: {e}")
        return "<h2>Error</h2><p>An error occurred while generating the analysis. Please try again later.</p>"

def create_similarity_table(semantic_field: List[str], similarities: List[float], n: int = 20) -> str:
    """Creates an HTML table of top N similar terms and their similarities."""
    table_html = "<table style='width:100%; border-collapse: collapse;'>"
    table_html += "<tr><th style='border:1px solid black; padding:8px;'>Term</th><th style='border:1px solid black; padding:8px;'>Similarity</th></tr>"
    for term, sim in sorted(zip(semantic_field, similarities), key=lambda x: abs(x[1]), reverse=True)[:n]:
        table_html += f"<tr><td style='border:1px solid black; padding:8px;'>{term}</td><td style='border:1px solid black; padding:8px;'>{sim:.4f}</td></tr>"
    table_html += "</table>"
    return table_html

def create_html_dashboard(fig: go.Figure, comprehensive_analysis: str, seo_table: str, semantic_table: str, output_dir: str):
    """Creates an improved HTML dashboard with Plotly figures, comprehensive analysis, and similarity tables."""
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SEO Content Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f0f0f0;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }}
                .plot {{
                    width: 100%;
                    height: 1200px;
                    margin-bottom: 20px;
                }}
                .analysis, .similarity-table {{
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                ul {{
                    padding-left: 20px;
                }}
                .scorecard {{
                    background-color: #e7f3fe;
                    border-left: 4px solid #2196F3;
                    padding: 10px;
                    margin-bottom: 10px;
                }}
                .recommendations {{
                    background-color: #e8f5e9;
                    border-left: 4px solid #4CAF50;
                    padding: 10px;
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>SEO Content Analysis Dashboard</h1>

                <div class="plot-container">  
                    <div id="plot" class="plot"></div>
                </div> 

                <div class="analysis">
                    <h2>Comprehensive Analysis and Recommendations</h2>
                    {comprehensive_analysis}
                </div>

                <div class="similarity-table">
                    <h2>Top 20 SEO Terms</h2>
                    {seo_table}
                </div>

                <div class="similarity-table">
                    <h2>Top 20 Semantic Terms</h2>
                    {semantic_table}
                </div>
            </div>

            <script>
                var plotlyData = {fig.to_json()};
                Plotly.newPlot('plot', plotlyData.data, plotlyData.layout);
            </script>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'seo_content_analysis_dashboard.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        logger.error(f"Error creating HTML dashboard: {e}")

def main():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'output_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

        content = read_content_from_file('dress.txt')
        if content is None:
            logger.error("Failed to read content. Exiting.")
            return

        keyword = "dress"

        semantic_field_seo = get_expanded_semantic_field(keyword, "seo", n=50, similarity_threshold=0.1)
        semantic_field_semantic = get_expanded_semantic_field(keyword, "semantic", n=50, similarity_threshold=0.1)

        content_vector = normalize_embedding(get_embedding(content))
        keyword_vector = normalize_embedding(get_embedding(keyword))
        
        neutral_term = "the"
        neutral_vector = normalize_embedding(get_embedding(neutral_term))
        
        field_vectors_seo = [normalize_embedding(get_embedding(term)) for term in semantic_field_seo]
        field_vectors_semantic = [normalize_embedding(get_embedding(term)) for term in semantic_field_semantic]
        
        keyword_diff_similarity = differential_similarity(content_vector, keyword_vector, neutral_vector)
        
        field_diff_similarities_seo = [differential_similarity(content_vector, vector, neutral_vector) for vector in field_vectors_seo]
        field_diff_similarities_semantic = [differential_similarity(content_vector, vector, neutral_vector) for vector in field_vectors_semantic]

        comprehensive_analysis = generate_comprehensive_analysis(
            keyword, content, keyword_vector, content_vector,
            semantic_field_seo, semantic_field_semantic,
            field_diff_similarities_seo, field_diff_similarities_semantic
        )

        fig = visualize_results_interactive(keyword, content_vector, keyword_vector,
                                            semantic_field_seo, semantic_field_semantic, 
                                            field_diff_similarities_seo, field_diff_similarities_semantic)

        seo_table = create_similarity_table(semantic_field_seo, field_diff_similarities_seo)
        semantic_table = create_similarity_table(semantic_field_semantic, field_diff_similarities_semantic)

        create_html_dashboard(fig, comprehensive_analysis, seo_table, semantic_table, output_dir)

        logger.info(f"Analysis completed. Results saved in '{output_dir}' directory.")

        # Save additional analysis results
        with open(os.path.join(output_dir, 'analysis_results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Differential similarity between content and '{keyword}' keyword: {keyword_diff_similarity:.4f}\n\n")
            f.write("SEO Semantic Field Analysis:\n")
            f.write("Term, Similarity\n")
            for term, sim in zip(semantic_field_seo, field_diff_similarities_seo):
                f.write(f"'{term}', {sim:.4f}\n")
            f.write("\nGeneral Semantic Field Analysis:\n")
            f.write("Term, Similarity\n")
            for term, sim in zip(semantic_field_semantic, field_diff_similarities_semantic):
                f.write(f"'{term}', {sim:.4f}\n")

        df_seo = pd.DataFrame({
            'Term': semantic_field_seo, 
            'Differential_Similarity': field_diff_similarities_seo
        })
        df_seo.to_csv(os.path.join(output_dir, 'seo_semantic_field_analysis.csv'), index=False)

        df_semantic = pd.DataFrame({
            'Term': semantic_field_semantic, 
            'Differential_Similarity': field_diff_similarities_semantic
        })
        df_semantic.to_csv(os.path.join(output_dir, 'general_semantic_field_analysis.csv'), index=False)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
