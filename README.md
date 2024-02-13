# AI and Data Science examples 🧠
Github Repository for the Streamlit "AI and Data Science examples" HEC Paris app.

Deploy the app locally using 
`streamlit run main_page.py`. 

The app currently has four use cases:
- Time Series Forecasting on Electrical Power Consumption 
- Sentiment Analysis on Customer reviews
- Recommendation system for movies


## Contributor guidelines
### Github

- Clone the repository locally (`git clone`)
- Create a virtual environment `python -m venv <venv_name>` ⇒ never push venv folder to github
- Install all the package dependencies using `pip install -r requirements.txt`
- **Create a branch** for each feature/use cases added
- Don’t push to the main branch, **create a pull request** to integrate changes to the repo
- Never push `.streamlit` folder to the repo
- Make sure your code is well documented (doc strings for functions,…)

### Streamlit

- To deploy the app locally, run on the terminal `streamlit run main_page.py`
- Create a page for each use case (in the `page`\ folder)
- Add datasets to the `data\` folder
- Add pretrained/saved models to the `pretrained_models` folder
- `.streamlit` : Add API keys, identifications… to a .yaml file (never pushed to the repo, add the “secrets” to the deployed app directly on the website)