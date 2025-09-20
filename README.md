# AI-Powered Job Application

This repository provides a suite of Python modules for automating and enhancing the job application process using AI-driven agents, resume parsing, and knowledge extraction. It is designed for experimentation and development of intelligent workflows for job seekers and HR automation.

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
    - Create and activate the environment using the provided `environment.yaml` file:
       ```powershell
       conda env create -f environment.yaml
       conda activate <env_name>
       ```

2. **Run the Agent**
   - Execute the main workflow:
     ```powershell
     python main.py
     ```

3. **Resume Parsing**
   - Place your resume PDF in the project folder and update the filename in `main.py` or `resume_parser.py` as needed.

4. **Knowledge Extraction & YAML Generation**
   - Use the provided modules to extract information and generate YAML files for further processing.

## Outputs
- Extracted data and agent results are saved as pickle files (`*.pkl`) and YAML files.
- Screenshots and visual outputs should be saved in a dedicated `screenshots/` directory (create if needed).

## Contributing
- Fork the repository and submit pull requests for improvements.
- Document any new modules or workflows in this README.

## License
Specify your license here (e.g., MIT, Apache 2.0).

## Contact
For questions or support, open an issue or contact the repository maintainer.
