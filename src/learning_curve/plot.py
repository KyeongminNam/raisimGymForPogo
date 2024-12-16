import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

ours_file = './original_Dec12_04-07-12.csv'
ab1_file = './no_priv_Dec13_03-55-14.csv'
ab2_file = 'MLP_Dec14_06-18-19.csv'

df_ours = pd.read_csv(ours_file)
df_ab1 = pd.read_csv(ab1_file)
df_ab2 = pd.read_csv(ab2_file)

fig, ax = plt.subplots(figsize=(10, 6))


ax.plot(df_ours['Step'].to_numpy(), df_ours['Value'].to_numpy(), color='red', label='Ours')
ax.plot(df_ab1['Step'].to_numpy(), df_ab1['Value'].to_numpy(), color='green', label='AB1')
ax.plot(df_ab2['Step'].to_numpy(), df_ab2['Value'].to_numpy(), color='blue', label='AB2')

ax.set_title('Average Reward')
ax.set_xlabel('Iterations')
ax.grid(True)
ax.legend(loc='upper right')
ax.set_xlim(-500, 24000)
ax.set_ylim(-10, 50)
ax.axvline(x=22000, linestyle='--', color='black', label='x=22000')


x1, x2, y1, y2 = 15000, 22000, 44, 49

# # Zoomed Inset
# axins = zoomed_inset_axes(ax, zoom=2, loc='lower right')
# axins.plot(df_ours['Step'].to_numpy(), df_ours['Value'].to_numpy(), color='red')
# axins.plot(df_ab2['Step'].to_numpy(), df_ab2['Value'].to_numpy(), color='blue')
#
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_xticks([])

# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='0.5')

plt.tight_layout()
plt.show()
