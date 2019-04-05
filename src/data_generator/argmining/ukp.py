
from data_generator.data_parser import ukp

all_topics = ["abortion", "cloning", "death_penalty", "gun_control",
                       "marijuana_legalization", "minimum_wage", "nuclear_energy", "school_uniforms"]

class DataLoader:
    def __init__(self, target_topic):

        self.test_topic = target_topic
        self.train_topics = list(set(all_topics) - {target_topic})
        self.all_data = {topic : ukp.load(topic) for topic in all_topics}
        self.labels = ["NoArgument", "Argument_for", "Argument_against"]


    def get_train_data(self):
        train_data = []
        for topic in self.train_topics:
            for entry in self.all_data[topic]:
                if entry['set'] == "train" :
                    x = entry['sentence']
                    y = self.labels.index(entry['annotation'])
                    train_data.append((x,y))
        return train_data

    def get_dev_data(self):
        dev_data = []
        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "val":
                x = entry['sentence']
                y = self.labels.index(entry['annotation'])
                dev_data.append((x,y))
        return dev_data


if __name__ == "__main__":
    d = DataLoader("abortion")
    d.get_train_data()
    d.get_dev_data()
