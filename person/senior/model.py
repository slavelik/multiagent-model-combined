from mesa import Model

from person.senior.agent import SeniorAgent


class SeniorModel(Model):
    def __init__(self, n_seniors, seed = None):
        super().__init__(seed=seed)
        self.current_hour = 0

        SeniorAgent.create_agents(model=self, n=n_seniors)

    def step(self):
        self.agents.shuffle_do("step")

        self.current_hour = (self.current_hour + 1)  % 24
