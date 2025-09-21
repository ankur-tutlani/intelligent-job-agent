# AI-Powered Job Application

AI and web‑agent powered Python tools for automating job applications, parsing resumes, and accelerating your job search.

## Technology & Models Used

- **LLMs**: All language models are accessed via OpenRouter and only free models are used, making the project completely free to use.
- **Embeddings**: The embedding model is from Huggingface and is also free to use.
- **Environment**: The code was tested in Python 3.11.13.


## Project Structure

- `main.py`: Main entry point for running the agent workflow.
- `resume_parser.py`: Extracts structured data from resumes (PDF supported).
- `extractors.py`: Logic for extracting information from various sources.
- `custom_multimodal.py`, `custom_textLLM.py`: Custom modules for multimodal and text-based LLM interactions.
- `prompt_builder.py`: Builds prompts for LLMs and agents.
- `knowledge_writer.py`, `generate_yaml.py`, `save_yaml.py`, `user_data_yaml.py`: Modules for knowledge extraction, YAML generation, and user data management.
- `web_agent.py`: Web automation agent for interacting with job portals.
- `config.py`: Configuration settings for the project.
- `knowledge.txt`, `objective.txt`: Text files containing domain knowledge and objectives.
- `user_data.pkl`: Pickle file for persisting user data. Its generated based on the information in resume.
- `SampleResume.pdf`: Example resume for testing.

## Usage

1. **Install Dependencies**
    - Ensure Python 3.8+ is installed.
    - Create and activate the environment using the provided `environment.yml` file:
       ```powershell
       conda env create -f environment.yml
       conda activate <env_name>
       ```

2. **Run the Agent**
   - Execute the main workflow:
     ```powershell
     python main.py
     ```

3. **Resume Parsing**
   - Place your resume PDF in the project folder and update the filename if needed.

4. **Knowledge Extraction & YAML Generation**
   - Use the provided modules to extract information and generate YAML files for further processing.

## Outputs
- Extracted data and agent results are saved as pickle files (`*.pkl`) and YAML files.
- Screenshots and visual outputs will be saved in a dedicated `screenshots/` directory (auto-created).

## Contributing
- Fork the repository and submit pull requests for improvements.
- Document any new modules or workflows in this README.

## License
This project is licensed under the MIT license.

## Contact
For questions or support, open an issue or contact the repository maintainer.
