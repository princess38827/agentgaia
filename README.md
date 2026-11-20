# agentgaia

GAIA benchmark evaluation and submission agent for **Hugging Face Agents Course — Unit 4 (Final Assignment)**.

## Quickstart
```bash
pip install -r requirements.txt
export HF_TOKEN="HFAKkYWSHQ6bKSlPSPz417V65QJ9mC6"
export HF_USERNAME="princess38827"
python app.py

@misc{mialon2023gaia,

      title={GAIA: a benchmark for General AI Assistants}, 

      author={Grégoire Mialon and Clémentine Fourrier and Craig Swift and Thomas Wolf and Yann LeCun and Thomas Scialom},

      year={2023},

      eprint={2311.12983},

      archivePrefix={arXiv},

      primaryClass={cs.CL}

}

cd Final_Assignment_Template
# overwrite app.py and requirements.txt with the files above
# create an empty Gradio Space on your account at hf.co/new-space, then:
git remote set-url origin https://huggingface.co/spaces/<AlanaSky>/<Agentgaia>
git add app.py requirements.txt && git commit -m "smolagents CodeAgent" && git push

