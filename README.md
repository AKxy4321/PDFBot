# PDFbot: A Chatbot that Interacts with PDFs

PDFbot is an interactive chatbot designed to communicate with the contents of a PDF. With PDFbot, you can engage in a conversation with any PDF file to extract information, ask questions, and gain insights. This project contains multiple modules to support various environments, including both local implementations and integration with OpenAI.

## Features
- **Chat with any PDF**: Ask questions about a PDF and receive intelligent responses.
- **Flexible Deployment**: Use it locally, with OpenAI, or with Llama 3.1 on Llama Cloud.
- **Gradio UI**: A user-friendly interface to interact with PDFbot locally.
- **Llama Cloud Parsing**: Uses **Llama Cloud** for robust PDF parsing and document processing.

## Project Structure
The project is organized into three major modules:

1. **PDFbot with OpenAI Integration**  
   - Located in a Jupyter Notebook (`PDFBot.ipynb`).
   - This module connects to OpenAI to process PDF content and generate responses based on your queries.

2. **Local PDFbot (Jupyter Notebook with Ollama)**  
   - Located in a Jupyter Notebook (`PDFBot_Ollama.ipynb`).
   - This version runs entirely on your local machine, processing the PDF without any external API calls.

3. **Local PDFbot with Gradio (Python Script)**  
   - Contained within the `PDFBot_Py/` folder.
   - Provides a local implementation using Gradio, allowing you to interact with PDFbot through a simple web interface.

## Installation

1. **Clone the Repository**
   ```bash
   git clone git@github.com:AKxy4321/PDFBot.git
   cd PDFBot

2. **Install Dependencies**

   * Make sure you have the necessary dependencies installed. You can use the `requirements.txt` file for easy setup:

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: The provided requirements are for the Ollama version. For OpenAI, check `PDFBot.ipynb`.)*

3. **Set Up API Keys**

   * **Llama Cloud (Llama 3.1)**:
     Sign up at [Llama Cloud](https://cloud.llamaindex.ai), create an API key, and set it in your environment:

     ```bash
     export LLAMA_CLOUD_API_KEY="your_api_key_here"
     ```

     Or, create a `.env` file with:

     ```
     LLAMA_CLOUD_API_KEY=your_api_key_here
     ```
   * **OpenAI (optional)**:
     If using the OpenAI module, set your `OPENAI_API_KEY`.

4. **Run Locally**

   * **For Local Notebook**: Open `PDFBot_Ollama.ipynb` and run the cells.
   * **For Gradio Interface**: Navigate to the `PDFBot_Py/` folder and run:

     ```bash
     python PDFBot.py
     ```

5. **Use OpenAI**

   * Open the `PDFBot.ipynb` file and follow the instructions to configure your OpenAI API key.

## How to Use

* Provide a PDF path and ask questions about its content.
* PDFbot will parse the document (using **Llama 3.1 via Llama Cloud** or your local setup) and provide precise answers.
* Use the Gradio interface for an easy-to-use, browser-based experience.

## Future Improvements

* Expand support for more file formats.
* Improve accuracy and context-awareness in conversations.
* Add additional customization options for local use and Gradio integration.
