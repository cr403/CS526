library(tidyverse) 

# Manual Weight Watcher's Points calculation based on formula from linear regression. These values are used throughout other visualizations. 
df_ww <- df %>% 
  mutate(Weight_Watchers_Pnts = ifelse(is.na(Weight_Watchers_Pnts), Sugars_g + Calories + Saturated_Fat_g - Protein_g, Weight_Watchers_Pnts))
