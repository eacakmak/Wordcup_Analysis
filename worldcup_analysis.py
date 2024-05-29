import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def load_data(file_path):
    """Verileri CSV dosyasından yükle."""
    data = pd.read_csv(file_path, delimiter=';')
    return data


def prepare_data(data):
    """Verileri her maç için toplam gol sayısını hesaplayacak şekilde hazırla."""
    # Veriyi pivot edelim ki her satır bir maç ve sütunlar ev sahibi ve misafir takımların golleri olsun
    data_home = data[data.index % 2 == 0].reset_index(drop=True)
    data_away = data[data.index % 2 == 1].reset_index(drop=True)

    data_home = data_home.rename(columns={'Team': 'Home_Team', 'Goal': 'Home_Goals'})
    data_away = data_away.rename(columns={'Team': 'Away_Team', 'Goal': 'Away_Goals'})

    data_combined = pd.concat([data_home, data_away[['Away_Team', 'Away_Goals']]], axis=1)
    data_combined['total_goals'] = data_combined['Home_Goals'] + data_combined['Away_Goals']

    return data_combined


def calculate_statistics(data):
    """İstatistiksel hesaplamaları yap."""
    # Ortalama ve Standart Sapma
    mean_goals = data['total_goals'].mean()
    std_dev_goals = data['total_goals'].std()

    # Hipotez Testi
    # H0: Maç başına düşen ortalama gol sayısı = 2
    # H1: Maç başına düşen ortalama gol sayısı > 2

    # Tek örneklem t-testi
    t_stat, p_value = stats.ttest_1samp(data['total_goals'], 2, alternative='greater')

    return mean_goals, std_dev_goals, t_stat, p_value


def plot_boxplot(data):
    """Kutu grafiğini çiz."""
    plt.boxplot(data['total_goals'])
    plt.title('Her Maçta Atılan Toplam Goller')
    plt.ylabel('Toplam Goller')
    plt.savefig('boxplot.png')  # Grafiği dosyaya kaydet
    plt.show()


def plot_histogram(data):
    """Histogramı çiz."""
    plt.hist(data['total_goals'], bins=range(0, max(data['total_goals']) + 1), edgecolor='black')
    plt.title('2022 Dünya Kupası Maçlarında Atılan Toplam Goller')
    plt.xlabel('Toplam Goller')
    plt.ylabel('Maç Sayısı')
    plt.savefig('histogram.png')  # Grafiği dosyaya kaydet
    plt.show()


def plot_test_value_critical_value(mean_goals, std_dev_goals, n, alpha=0.05):
    """Test ve kritik değerlerin grafiğini çiz."""
    z_critical = stats.norm.ppf(1 - alpha)  # One-tailed test
    z_value = (mean_goals - 2) / (std_dev_goals / np.sqrt(n))

    x = np.linspace(-3, 3, 1000)
    y = stats.norm.pdf(x, 0, 1)

    plt.plot(x, y, label='Normal Distribution')
    plt.axvline(x=z_value, color='red', linestyle='-', linewidth=2, label=f'Test Value = {z_value:.4f}')
    plt.axvline(x=z_critical, color='green', linestyle='-', linewidth=2, label=f'Critical Value = {z_critical:.3f}')
    plt.fill_between(x, 0, y, where=(x > z_critical), color='green', alpha=0.3, label='Reject Region')
    plt.fill_between(x, 0, y, where=(x <= z_critical), color='blue', alpha=0.1, label='Do Not Reject Region')

    plt.title('Test Value and Critical Value')
    plt.xlabel('z')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig('test_critical_value.png')  # Grafiği dosyaya kaydet
    plt.show()


def main():
    # add file path
    file_path = r'worldcup matches 2022.csv'  
    data = load_data(file_path)

    # İlk birkaç satırı ve sütun adlarını görüntüleyin
    print(data.head())
    print(data.columns)

    data_prepared = prepare_data(data)

    mean_goals, std_dev_goals, t_stat, p_value = calculate_statistics(data_prepared)

    print(f"Ortalama Gol Sayısı: {mean_goals}")
    print(f"Standart Sapma: {std_dev_goals}")
    print(f"t-İstatistiği: {t_stat}")
    print(f"p-Değeri: {p_value}")

    plot_boxplot(data_prepared)
    plot_histogram(data_prepared)

    # Test ve kritik değer grafiğini çiz
    n = len(data_prepared)
    plot_test_value_critical_value(mean_goals, std_dev_goals, n)


if __name__ == "__main__":
    main()
