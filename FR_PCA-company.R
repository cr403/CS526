library(tidyverse) 
library(factoextra) 
library(stats) 

# Build PCA 
```{r}
# For the PCA, I'm going to use all variables except calories from fat and weight watchers score 
# since these are both linearly derived from other variables (e.g., total fat/calories, sugar, 
# saturated fat, and protein)

pca_att <- c("Calories", "Total_Fat_g", "Saturated_Fat_g", 
                  "Trans_Fat_g", "Cholesterol_mg", "Sodium_mg", "Carbs_g", 
                  "Fiber_g", "Sugars_g", "Protein_g") 

# separate numeric attributes from metadata 
df_num <- df_ww %>%
  select(pca_att) 
df_meta <- df_ww %>%
  select(Company, Item) 

# identify complete rows (no missing numerical values) 
complete <- complete.cases(df_num) 

# remove incomplete rows 
df_num_clean <- df_num[complete, ]
df_meta_clean <- df_meta[complete, ]

# transform numerical data 
df_scaled <- df_num_clean %>%
  log1p() %>% # reduce skewedness 
  scale()  # ensures equal feature contribution 

# Perform PCA
pca_result <- prcomp(df_scaled, center = TRUE, scale. = TRUE)

# Scree plot: Variance explained by each principal component
fviz_eig(pca_result)

# Biplot
p <- fviz_pca_biplot(pca_result, 
                      label = "var",  # Show variable labels
                      repel = TRUE,    # Avoid overlapping labels
                      col.var = "blue",  # Variables (nutrients) in blue
                      col.ind = df_meta_clean$Company,  # Color individuals (items) by Company
                      addEllipses = FALSE,  # Optional: Add confidence ellipses for each company
                      legend.title = "Company", 
                     geom = "point", 
                     pointshape = 16)

# Add colors 
p + scale_color_manual(values = customPalette)
```
