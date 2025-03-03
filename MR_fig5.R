```{r}
# Choose attributes here 
pca_att <- c("Calories", "Sodium_mg", "Cholesterol_mg", "Sugars_g")

# separate numeric attributes from metadata 
df_num <- df %>%
  select(pca_att) 
df_meta <- df %>%
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

# Optional: Scree plot - Variance explained by each principal component
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
