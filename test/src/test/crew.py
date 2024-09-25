from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import requests

# Custom RAG LLM query function
def query_rag_llm(query):
    api_url = "http://127.0.0.1:8000/ask/"
    
    # Ensure the input query is a string
    if not isinstance(query, str):
        query = str(query)
    
    payload = {"query": query}
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("answer", "No answer returned from API")
    else:
        return f"Error: {response.status_code}, {response.text}"


# Wrapper class for the LLM
class LLMWrapper:
    def __init__(self, query_function):
        self.query_function = query_function

    def bind(self, stop=None):
        return self.query_function


@CrewBase
class TestCrew:
    """Test crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            llm=LLMWrapper(query_rag_llm),  # Use the LLM wrapper
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            llm=LLMWrapper(query_rag_llm),  # Use the LLM wrapper
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            agent=self.reporting_analyst(),
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Test crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

def main():
    crew_instance = TestCrew()
    crew_instance.run()

if __name__ == "__main__":
    main()