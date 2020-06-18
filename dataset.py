import yaml

class Dataset:
    def __init__(self, features_file_path):
        self.initialize()
        self.load_dataset(features_file_path)
        self.size = len(self.features)
    #enddef

    def load_dataset(self, features_file_path):
        features_file = open(features_file_path, 'r')
        lines_dict = yaml.load(features_file)
        for key in lines_dict:
            self.features.append(lines_dict[key])
            if key < 40:
                self.caterpillar_moth_features.append(lines_dict[key])
                self.caterpillar_moth_labels.append(0)
                self.labels.append(-1)
                self.sub_species_labels.append(0)
            elif key >= 40 and key < 76:
                self.tiger_moth_features.append(lines_dict[key])
                self.tiger_moth_labels.append(1)
                self.labels.append(-1)
                self.sub_species_labels.append(1)
            elif key >= 76 and key < 116:
                self.ringlet_butterfly_features.append(lines_dict[key])
                self.ringlet_butterfly_labels.append(2)
                self.labels.append(1)
                self.sub_species_labels.append(2)
            else:
                self.peacock_butterfly_features.append(lines_dict[key])
                self.peacock_butterfly_labels.append(3)
                self.labels.append(1)
                self.sub_species_labels.append(3)
        return
    #enddef

    def initialize(self):
        self.features = []
        self.labels = []
        self.sub_species_labels = []
        self.caterpillar_moth_features = []
        self.caterpillar_moth_labels = []
        self.tiger_moth_features = []
        self.tiger_moth_labels = []
        self.ringlet_butterfly_features = []
        self.ringlet_butterfly_labels = []
        self.peacock_butterfly_features = []
        self.peacock_butterfly_labels = []
    #enddef


if __name__ == "__main__":
    pass
