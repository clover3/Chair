import matplotlib.pyplot as plt

from cache import load_pickle_from
from cpath import output_path
from misc_lib import path_join, average
from trainer_v2.per_project.transparency.misc_common import sota_retrieval_methods


def draw_chart(labels, values):
    # Creating the bar chart
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Added letters', fontsize=12)  # Change font size here
    plt.ylabel('Score changes', fontsize=12)  # Change font size here
    plt.xticks(rotation=0)
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(x, '.2f')))  # Two decimal places

    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    return plt

def value_example():

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
    values = [
        -0.00820500080932876, 0.1310710813887343, 0.12529361468113348, 0.1347097317258755, 0.1130408585190535,
        0.12341313587572285, 0.11546545212497254, 0.16372339276340372, 0.1285384099878475, 0.1429567006593217,
        0.15901448331520468, 0.15293433078510796, 0.08745833568111389, 0.12658730580540237, 0.11090059040073387,
        0.17859506108770204, 0.14398081966502937, 0.1471543980617366, 0.05467552363039729, 0.1542828060433774,
        0.0991452518337501, 0.134283896740117, 0.12002501041886811, 0.15576012090532604, 0.15928329867933563,
        0.12278295811005457
    ]

def main():
    # Data
    a_to_z = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

    for method_name in sota_retrieval_methods:
        save_path = path_join(output_path, "mmp", "append_test", method_name)
        d: dict[str, list[float]] = load_pickle_from(save_path)
        mean_drop = [average(d[c]) for c in a_to_z]
        print(method_name)
        print(mean_drop)
        change_value = [-t for t in mean_drop]
        plt = draw_chart(a_to_z, change_value)
        img_save_path = path_join(output_path, "mmp", "append_test", method_name + "2.png")
        plt.savefig(img_save_path)
        plt.clf()
        break




if __name__ == "__main__":
    main()