import matplotlib.pyplot as plt


def plot_correlation_matrix(corrs, corrs_avg, max_n_components, figsize=(22, 45)):
    """
    Plot the correlation matrix.

    Parameters:
        corrs (pd.DataFrame): DataFrame containing correlations.
        corrs_avg (pd.DataFrame): DataFrame containing weighted average correlations.
        max_n_components (int): Maximum number of components for the subplot grid.
        figsize (tuple, optional): Size of the figure. Defaults to (22, 45).
    """

    # Determine the number of rows and columns for the subplot grid
    n_rows = len(set(corrs.droplevel(3, axis=1).columns))
    n_cols = max_n_components + 1

    # Create a subplot grid with specified size and shared y-axis
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, sharey='row')

    # Plot correlations
    pos = -1
    for idx, key_dim in enumerate(list(corrs.keys())):
        values = corrs[key_dim]
        categories = values.index
        if key_dim[-1] == 0:
            pos += 1
        ax[pos][key_dim[-1]+1].barh(categories, values)

    # Plot average correlations
    for idx, key_dim in enumerate(list(corrs_avg.keys())):
        values = corrs_avg[key_dim]
        categories = values.index
        ax[idx][0].barh(categories, values)

    # Add column labels
    col_labels = ['Avg.', *[str(x) for x in range(1, max_n_components + 1)]]
    for ax_, col in zip(ax[0], col_labels):
        ax_.annotate(col, xy=(0.5, 1), xytext=(0, 20),
                     xycoords='axes fraction', textcoords='offset points',
                     size=30, ha='center', va='baseline')

    # Add row labels
    row_labels = [x[1] + '\n' + x[2] for x in corrs.keys().to_list()]
    row_labels = [x for i, x in enumerate(
        row_labels) if x not in row_labels[:i]]
    for ax_, row in zip(ax[:, 0], row_labels):
        ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - 75, 0),
                     xycoords=ax_.yaxis.label, textcoords='offset points',
                     size=20, ha='center', va='center')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.0)

    # Set the range of the x-axis for all plots
    for ax_row in ax:
        xlims = [x.get_xlim() for x in ax_row[1:]]
        xlims = [x for x in xlims if x != (0.0, 1.0)]
        xlim = max(xlims, key=lambda t: abs(t[0] - t[1]))
        for ax_ in ax_row[1:]:
            ax_.set_xlim(xlim)

    plt.tight_layout()  # Adjust subplot layout to prevent overlapping
    plt.show()
