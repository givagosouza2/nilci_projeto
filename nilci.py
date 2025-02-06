import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import vonmises
import pandas as pd
import warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Funções para Estatística Circular
# =============================================================================


def circular_statistics(angles):
    """
    Calcula estatísticas descritivas para dados circulares.

    Parâmetros:
        angles : array-like
            Vetor de ângulos em radianos (no intervalo [0, 2π)).

    Retorna:
        mean_angle : float
            Média angular (em radianos, no intervalo [0, 2π)).
        R : float
            Comprimento do vetor resultante.
        circ_variance : float
            Variância circular (1 - R).
        circ_std : float
            Desvio padrão circular (em radianos).
        circ_median : float
            Mediana circular (em radianos).
        circ_skewness : float
            Assimetria circular.
        circ_kurtosis : float
            Curtose circular.
    """
    n = len(angles)
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mean_angle = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
    R = np.sqrt(sin_sum**2 + cos_sum**2) / n
    circ_variance = 1 - R
    circ_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.inf

    # Mediana Circular
    def circular_distance_sum(median):
        return np.sum(np.abs(np.angle(np.exp(1j * (angles - median)), deg=False)))
    circ_median = minimize(circular_distance_sum,
                           x0=np.median(angles)).x[0] % (2 * np.pi)

    # Assimetria Circular
    circ_skewness = np.mean(np.sin(2 * (angles - mean_angle)))

    # Curtose Circular
    circ_kurtosis = np.mean(np.cos(2 * (angles - mean_angle)))

    return mean_angle, R, circ_variance, circ_std, circ_median, circ_skewness, circ_kurtosis


def rayleigh_test(angles):
    """
    Realiza o teste de Rayleigh para verificar a hipótese de uniformidade em dados circulares.

    Parâmetros:
        angles : array-like
            Vetor de ângulos em radianos.

    Retorna:
        z : float
            Estatística de teste (n * R²).
        p_value : float
            Valor-p aproximado.
    """
    n = len(angles)
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    R = np.sqrt(sin_sum**2 + cos_sum**2)
    z = (R**2) / n
    p_value = np.exp(-z) * (1 + (2*z - z**2) / (4*n) -
                            (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    return z, p_value

# =============================================================================
# Implementação do Rao's Spacing Test
# =============================================================================


def rao_spacing_test(angles, n_sim=1000, random_state=42):
    """
    Realiza o teste de Rao's Spacing para avaliar a uniformidade de dados circulares.

    Parâmetros:
        angles : array-like
            Vetor de ângulos em radianos.
        n_sim : int, opcional
            Número de simulações para estimar o valor-p (Monte Carlo).
        random_state : int, opcional
            Semente para reprodutibilidade das simulações.

    Retorna:
        U_obs : float
            Estatístico de Rao observado.
        p_value : float
            Valor-p estimado.
    """
    rng = np.random.default_rng(random_state)
    n = len(angles)
    # Converter os ângulos para graus para facilitar o cálculo
    angles_deg = np.degrees(angles) % 360
    angles_ord = np.sort(angles_deg)

    # Cálculo dos espaçamentos entre ângulos ordenados (incluindo o fechamento do círculo)
    spacings = np.diff(np.concatenate((angles_ord, [angles_ord[0] + 360])))
    expected_spacing = 360 / n
    U_obs = 0.5 * np.sum(np.abs(spacings - expected_spacing))

    # Estimação do p-valor via simulação
    U_sim = []
    for _ in range(n_sim):
        sim_angles = rng.uniform(0, 360, n)
        sim_angles = np.sort(sim_angles)
        sim_spacings = np.diff(np.concatenate(
            (sim_angles, [sim_angles[0] + 360])))
        U_sim_val = 0.5 * np.sum(np.abs(sim_spacings - expected_spacing))
        U_sim.append(U_sim_val)
    U_sim = np.array(U_sim)
    # p-valor: proporção de simulações com U_sim >= U_obs
    p_value = np.mean(U_sim >= U_obs)
    return U_obs, p_value

# =============================================================================
# Estimação de Densidade para Dados Circulares (KDE com kernel von Mises)
# =============================================================================


def circular_kde(angles, grid_points=360, kappa=4):
    """
    Estima a densidade de probabilidade para dados circulares usando um kernel von Mises.

    Parâmetros:
        angles : array-like
            Vetor de ângulos em radianos.
        grid_points : int, opcional
            Número de pontos na grade para avaliação da densidade.
        kappa : float, opcional
            Parâmetro de concentração da distribuição von Mises.

    Retorna:
        grid : array
            Vetor de ângulos (em radianos) no intervalo [0, 2π].
        densities : array
            Densidade estimada para cada ponto da grade.
    """
    grid = np.linspace(0, 2*np.pi, grid_points)
    densities = np.zeros_like(grid)
    n = len(angles)
    for angle in angles:
        densities += vonmises.pdf(grid, kappa, loc=angle)
    densities /= n
    return grid, densities

# =============================================================================
# Plotagem: Rose Diagram (Histograma Circular) com a linha do vetor médio
# =============================================================================


def plot_rose(angles, bins=16, mean_angle=None, R_val=None):
    """
    Plota um diagrama de rosca (rose plot) para os dados circulares e, opcionalmente,
    sobrepõe uma linha vermelha indicando o vetor médio.

    Parâmetros:
        angles : array-like
            Vetor de ângulos em radianos.
        bins : int, opcional
            Número de bins do histograma.
        mean_angle : float, opcional
            Média angular (em radianos). Se fornecido, será desenhada a linha do vetor médio.
        R_val : float, opcional
            Comprimento do vetor resultante. Se fornecido, é utilizado para dimensionar a linha.
            Em geral, R_val varia entre 0 e 1.
    """
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    counts, bin_edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi))
    widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + widths/2
    bars = ax.bar(bin_centers, counts, width=widths, bottom=0.0,
                  align='center', alpha=0.5, edgecolor='k')

    # Se a média e R forem fornecidos, sobrepõe uma linha vermelha para o vetor médio.
    if (mean_angle is not None) and (R_val is not None):
        # Para o Rose Diagram, usamos o máximo das contagens para definir o comprimento da seta.
        max_count = np.max(counts)
        arrow_length = R_val * max_count
        # Desenha uma linha (vetor) a partir do centro até (arrow_length, mean_angle)
        ax.plot([mean_angle, mean_angle], [0, arrow_length],
                color='red', lw=3, label='Vetor Médio')
        ax.legend(loc='upper right')

    ax.set_title("Rose Diagram")
    return fig

# =============================================================================
# Plotagem: Estimação de Densidade Circular (KDE)
# =============================================================================


def plot_kde(angles, grid_points=360, kappa=4):
    """
    Plota a densidade de probabilidade estimada para dados circulares.

    Parâmetros:
        angles : array-like
            Vetor de ângulos em radianos.
        grid_points : int, opcional
            Número de pontos na grade para avaliação da densidade.
        kappa : float, opcional
            Parâmetro de concentração da distribuição von Mises.
    """
    grid, densities = circular_kde(
        angles, grid_points=grid_points, kappa=kappa)
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.plot(grid, densities, label='KDE (von Mises)', color='r')
    ax.set_title("Densidade de Probabilidade Circular Estimada")
    ax.legend()
    return fig

# =============================================================================
# Plotagem: Pontos dos ângulos na Circunferência da Unidade
# =============================================================================


def plot_angles_on_unit_circle(angles):
    """
    Plota os ângulos como pontos sobre a circunferência unitária.

    Parâmetros:
        angles : array-like
            Vetor de ângulos em radianos.
    """
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    # Todos os pontos na circunferência possuem raio 1.
    r = np.ones_like(angles)
    # Plota os pontos como círculos azuis.
    ax.scatter(angles, r, color='blue', s=50, label="Pontos de Ângulo")
    # Opcional: desenha a circunferência unitária
    theta = np.linspace(0, 2*np.pi, 360)
    ax.plot(theta, np.ones_like(theta), color='gray', ls='--', lw=1)
    ax.set_title("Pontos de Ângulo na Circunferência Unitária")
    ax.legend(loc='upper right')
    return fig

# =============================================================================
# Função de Análise Circular Integrada ao Streamlit
# =============================================================================


def analise_circular(angles, kde_kappa=4, rose_bins=16):
    # Cálculo das estatísticas circulares
    mean_angle, R, circ_variance, circ_std, circ_median, circ_skewness, circ_kurtosis = circular_statistics(
        angles)
    z, p_rayleigh = rayleigh_test(angles)
    U_rao, p_rao = rao_spacing_test(angles, n_sim=1000)

    st.subheader("Estatísticas Circulares Descritivas")
    st.write(
        f"**Média Angular:** {mean_angle:.3f} rad ({np.degrees(mean_angle):.1f}°)")
    st.write(
        f"**Mediana Circular:** {circ_median:.3f} rad ({np.degrees(circ_median):.1f}°)")
    st.write(f"**Assimetria Circular:** {circ_skewness:.3f}")
    st.write(f"**Curtose Circular:** {circ_kurtosis:.3f}")
    st.write(f"**Comprimento do Vetor Resultante (R):** {R:.3f}")
    st.write(f"**Variância Circular:** {circ_variance:.3f}")
    st.write(
        f"**Desvio Padrão Circular:** {circ_std:.3f} rad ({np.degrees(circ_std):.1f}°)")

    st.subheader("Teste de Uniformidade")
    st.write("**Teste de Rayleigh:**")
    st.write(f"  Estatística z: {z:.3f}")
    st.write(f"  Valor-p: {p_rayleigh:.3f}")
    if p_rayleigh < 0.05:
        st.write("  → Rejeita a hipótese de distribuição uniforme.")
    else:
        st.write("  → Não rejeita a hipótese de distribuição uniforme.")

    st.write("**Rao's Spacing Test:**")
    st.write(f"  Estatístico U: {U_rao:.3f}")
    st.write(f"  Valor-p (estimado via Monte Carlo): {p_rao:.3f}")
    if p_rao < 0.05:
        st.write("  → Rejeita a hipótese de distribuição uniforme.")
    else:
        st.write("  → Não rejeita a hipótese de distribuição uniforme.")

    # Plot do Rose Diagram com a linha do vetor médio
    st.subheader("Rose Diagram com Vetor Médio")
    fig_rose = plot_rose(angles, bins=rose_bins,
                         mean_angle=mean_angle, R_val=R)
    st.pyplot(fig_rose)

    # Plot da Estimação de Densidade Circular (KDE)
    st.subheader("Densidade de Probabilidade Circular (KDE)")
    fig_kde = plot_kde(angles, grid_points=360, kappa=kde_kappa)
    st.pyplot(fig_kde)

    # Plot dos ângulos na circunferência unitária
    st.subheader("Pontos de Ângulo na Circunferência Unitária")
    fig_unit = plot_angles_on_unit_circle(angles)
    st.pyplot(fig_unit)

# =============================================================================
# Função para Cálculo dos Ângulos a partir das Coordenadas
# =============================================================================


def calculate_angles(x_coords, y_coords):
    """
    Cálculo vetorizado de ângulos entre pontos consecutivos com validação avançada.

    Converte os ângulos para o intervalo [0, 2π) para manter a consistência.
    """
    try:
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # Identificar pontos consecutivos não duplicados
        valid_mask = (dx != 0) | (dy != 0)
        if not np.any(valid_mask):
            st.error("Todos os pontos consecutivos são duplicados!")
            return None

        angles = np.arctan2(dy[valid_mask], dx[valid_mask])
        # Converte para o intervalo [0, 2π)
        angles = angles % (2 * np.pi)
        return angles

    except Exception as e:
        st.error(f"Erro no cálculo de ângulos: {str(e)}")
        return None

# =============================================================================
# Interface Interativa com Streamlit
# =============================================================================


def main():
    st.set_page_config(page_title="Análise Espacial Avançada", layout="wide")
    st.title("🌍 Análise de Coordenadas Espaciais com Estatística Circular")

    # Widget de upload de arquivo
    with st.sidebar:
        st.header("Entrada de Dados")
        uploaded_file = st.file_uploader(
            "Selecione o arquivo do acelerômetro", type=["txt", "csv"],
            accept_multiple_files=False
        )
        st.markdown("""
        **Requisitos de Formato:**
        - Valores separados por vírgula ou espaço
        - Mínimo 3 colunas numéricas
        - Coluna 0: Tempo (opcional)
        - Coluna 1: Coordenada X
        - Coluna 2: Coordenada Y
        - Sem cabeçalho ou caracteres especiais
        """)

    # Processamento dos dados
    if uploaded_file is not None:
        # Leitura e validação dos dados
        try:
            df = pd.read_csv(uploaded_file, sep=",", header=0)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            st.stop()

        if df.shape[1] < 3:
            st.error("Arquivo insuficiente: requer pelo menos 3 colunas")
            st.stop()

        # Converter as colunas X e Y para valores numéricos
        x = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        y = pd.to_numeric(df.iloc[:, 2], errors='coerce')

        if x.isna().any() or y.isna().any():
            st.error("Dados não numéricos nas colunas X/Y")
            st.stop()

        # Cálculo dos ângulos entre pontos consecutivos
        angles = calculate_angles(x.values, y.values)
        if angles is None:
            st.stop()

        st.sidebar.header("Parâmetros dos Gráficos")
        rose_bins = st.sidebar.slider(
            "Número de bins (Rose Diagram)", 8, 32, 16)
        kde_kappa = st.sidebar.slider("Kappa para KDE", 1, 20, 4)

        analise_circular(angles, kde_kappa=kde_kappa, rose_bins=rose_bins)
        col1, col2, col3 = st.columns(3)
        with col2:
            fig, ax1 = plt.subplots(figsize=(18, 6))
            # Gráfico espacial
            ax1.plot(x, y, 'r--', lw=1, alpha=0.3)
            ax1.scatter(x, y, c='b', alpha=0.7, s=50)
            ax1.set_title("Trajetória Espacial")
            ax1.set_xlabel("Coordenada X")
            ax1.set_ylabel("Coordenada Y")
            ax1.set_xlim(0, 1440)
            ax1.set_ylim(0, 2730)
            ax1.set_aspect('equal')
            ax1.grid(True)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
