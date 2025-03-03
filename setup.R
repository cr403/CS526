# Load libraries 
```{r}
library(tidyverse)
library(factoextra)
```

# Read data 
```{r}
df_raw <- read_csv("FastFoodNutritionMenuV3.csv") 
```

# Minor alteration of attributes for visualization
```{r}
# Remove spaces from colnames for ease of use 
colnames(df_raw) <- colnames(df_raw) %>% 
  str_replace_all("\n", " ") %>%
  trimws() %>%
  str_squish() %>%
  str_replace_all(" ", "_") %>%
  str_replace_all("\\(", "") %>%
  str_replace_all("\\)", "")

df_raw <- df_raw %>% 
  mutate(across(-c("Company", "Item"), as.numeric),
         Company = trimws(gsub("’", "'", Company)), # Remove white spaces, make apostrophes straight instead of curly
         Company = factor(Company, levels = c(
                                              "Burger King", 
                                              "KFC", 
                                              "McDonald's", 
                                              "Pizza Hut", 
                                              "Taco Bell", 
                                              "Wendy's")))

# Working data frame for the analysis 
df <- df_raw
```

# Assign color palette for all visualizations 
```{r}
customPalette <- c("Burger King" = "#ef476f",
                   "KFC" = "#f78c6b",
                   "McDonald's" = "#ffd166",
                   "Pizza Hut" = "#06d6a0",
                   "Taco Bell" = "#118ab2",
                   "Wendy's" = "#073b4c")
```
