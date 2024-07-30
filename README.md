# SEO Content Analyzer

## Description

The SEO Content Analyzer is a powerful Python script that performs semantic analysis on content to provide valuable insights for search engine optimization (SEO). It uses advanced natural language processing techniques, including sentence transformers and OpenAI's GPT model, to analyze the relevance of content to specific keywords and generate comprehensive SEO recommendations.

## Features

- Semantic field analysis for both SEO and general context
- Interactive visualization of semantic similarities
- Comprehensive SEO analysis and recommendations
- HTML dashboard generation with Plotly graphs
- Similarity tables for top SEO and semantic terms
- CSV exports of analysis results

## Requirements

- Python 3.7+
- OpenAI API key
- Required Python packages (install via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - sentence-transformers
  - plotly
  - umap-learn
  - scikit-learn
  - python-dotenv
  - openai

## Setup

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Prepare your content:
   - Create a text file with your content (e.g., `your_content.txt`)
   - Place the file in the project root directory

2. Modify the script:
   - Open `robust-checker.py` in a text editor
   - Locate the `main()` function
   - Update the following lines:
     ```python
     content = read_content_from_file('your_content.txt')  # Change 'dress.txt' to your file name
     keyword = "your_keyword"  # Change "dress" to your desired keyword
     ```

3. Run the script: `python robust-checker.py`

4. Check the `output_YYYYMMDD_HHMMSS` directory for results:
   - `seo_content_analysis_dashboard.html`: Interactive dashboard
   - `analysis_results.txt`: Detailed analysis results
   - `seo_semantic_field_analysis.csv` and `general_semantic_field_analysis.csv`: CSV exports of semantic field analyses

By following these steps, you can analyze any content file with any keyword of your choice.

## OpenAI API Usage and Limitations

**Important:** This script uses the OpenAI API, which has usage limits and associated costs. Please be aware of the following:

1. **API Key**: You need an OpenAI API key to use this script. If you don't have one, you can obtain it from the [OpenAI website](https://openai.com).

2. **Costs**: Using the OpenAI API incurs costs based on the number of tokens processed. The script uses GPT-4, which is more expensive than other models. Monitor your usage to avoid unexpected charges.

3. **Rate Limits**: OpenAI imposes rate limits on API requests. If you're analyzing a large volume of content, you may need to implement rate limiting in the script to avoid hitting these limits.

4. **Content Guidelines**: Ensure that the content you're analyzing complies with OpenAI's content policies. Avoid using the API for generating or analyzing harmful or explicit content.

5. **Token Limits**: GPT-4 has a maximum token limit per request. Very long pieces of content may need to be split into smaller chunks for analysis.

6. **API Changes**: OpenAI may update their API or models, which could affect this script's functionality. Keep an eye on OpenAI's documentation for any changes.

7. **Data Privacy**: Be cautious about the data you send to the OpenAI API. Avoid sending sensitive or personal information.

8. **Billing**: Set up billing alerts in your OpenAI account to avoid unexpected charges.

It's recommended to start with small tests to understand the API usage and costs before running large-scale analyses. Always refer to the [OpenAI API documentation](https://platform.openai.com/docs/introduction) for the most up-to-date information on usage, limits, and best practices.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
