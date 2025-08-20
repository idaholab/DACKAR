
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plotWordCloud(text, save=False, name="wordcloud"):
  """plot word cloud for given text

  Args:
      text (str): text of words to be plotted
      save (bool, optional): save plot to file if True. Defaults to False.
      name (str, optional): name for the plot file. Defaults to wordcloud.
  """
  wc = WordCloud(max_font_size=40, background_color="white").generate(text)
  plt.figure()
  plt.imshow(wc, interpolation="bilinear")
  plt.axis("off")
  if save:
    plt.savefig(f"{name}.png")
  plt.close()

