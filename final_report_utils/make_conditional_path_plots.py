import numpy as np
import matplotlib.pyplot as plt

# Define the functions for each group
def kappa_group1(x):
    kappa_1 = x
    kappa_2 = np.ones_like(x)
    return kappa_1, kappa_2

def kappa_group2(x):
    kappa_1 = x**2
    kappa_2 = np.ones_like(x)
    return kappa_1, kappa_2

def kappa_group3(x):
    kappa_1 = 2 * x**3 - 3 * x**2 + 2 * x
    kappa_2 = np.ones_like(x)
    return kappa_1, kappa_2

def kappa_group4(x):
    kappa_1 = x**2
    kappa_2 = x
    kappa_3 = np.ones_like(x)
    return kappa_1, kappa_2, kappa_3

# Create a function to plot and shade the areas
def plot_group(ax, x, kappa_functions, path_type):
    kappa_1 = kappa_functions[0](x)
    ax.fill_between(x, 0, kappa_1, color='red', alpha=0.5, label=r'$\delta_{x_1}$')
    ax.plot(x, kappa_1, color='red')

    if len(kappa_functions) > 2:
        
        kappa_2 = kappa_functions[1](x)
        label = r'noise'
        ax.fill_between(x, kappa_1, kappa_2, color='yellow', alpha=0.5, label=label)
        ax.plot(x, kappa_2, color='yellow')

        kappa_3 = kappa_functions[2](x)
        ax.fill_between(x, kappa_2, kappa_3, color='blue', alpha=0.5, label=r'$\delta_{x_0}$')
        ax.plot(x, kappa_3, color='blue')
    elif len(kappa_functions) > 1:
        kappa_2 = kappa_functions[1](x)
        label = r'$\delta_{x_0}$'
        ax.fill_between(x, kappa_1, kappa_2, color='blue', alpha=0.5, label=label)
        ax.plot(x, kappa_2, color='blue')


    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)  # Set y-axis limit to stop at 1
    ax.set_title(f'{path_type}', size='14')
    ax.set_xlabel('t', size='14')
    ax.set_ylabel(r'cumulative $\kappa^j(x)$', size='14')

# Define the x range
x = np.linspace(0, 1, 500)

# Create a single figure with 4 subplots side by side
fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

# Plot each group in its respective subplot
plot_group(axs[0], x, (lambda x: kappa_group1(x)[0], lambda x: kappa_group1(x)[1]), 'linear')
plot_group(axs[1], x, (lambda x: kappa_group2(x)[0], lambda x: kappa_group2(x)[1]), 'quadratic')
plot_group(axs[2], x, (lambda x: kappa_group3(x)[0], lambda x: kappa_group3(x)[1]), 'cubic')
plot_group(axs[3], x, (lambda x: kappa_group4(x)[0], lambda x: kappa_group4(x)[1], lambda x: kappa_group4(x)[2]), 'quadratic with noise')

# Combine the legends from all subplots into one
handles, labels = axs[-1].get_legend_handles_labels()  # Get handles and labels from the last subplot
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.04, 0.76), bbox_transform=plt.gcf().transFigure, prop={'size': 15})

# Adjust layout so that the legend fits well
plt.subplots_adjust(right=0.85)  # Make space on the right for the legend

# Save the combined figure as a PNG file
plt.savefig('combined_plots.png', dpi=300, bbox_inches='tight')

plt.show()
