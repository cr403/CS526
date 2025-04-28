df_pca_k <- df_pca %>%
  mutate(id = paste(Company, Item, row_number(.), sep = "-")) %>%
  select(-c(Company, Item, Weight_Watchers_Pnts)) %>%
  left_join(df_clr_meta_id, by = "id") %>% 
  column_to_rownames(var = "id") 

df_pca_k_num <- df_pca_k %>%
  select(pca_att) 

df_pca_k_meta <- df_pca_k %>%
  select(-pca_att)

# transform numerical data 
df_scaled <- df_pca_k_num %>%
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
                      col.ind = df_pca_k_meta$k_cluster,  # Color individuals (items) by Company
                      addEllipses = FALSE,  # Optional: Add confidence ellipses for each company
                      legend.title = "Cluster", 
                     geom = "point", 
                     pointshape = 16)
