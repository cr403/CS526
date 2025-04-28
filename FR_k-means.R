```{r}
set.seed(123)  # for reproducibility

# Run k-means clustering on CLR-transformed data
k_result <- kmeans(df_clr, centers = 4, nstart = 25)  # change centers = k as needed
```

Add cluster information to metadata
```{r}
df_clr_num_raw2 <- df_clr_num_raw %>%
  rownames_to_column(var = "id")

df_clr_meta$k_cluster <- factor(k_result$cluster) 

df_clr_data <- df_clr_meta %>%
  rownames_to_column(var = "id") %>%
  left_join(df_clr_num_raw2, by = "id") %>%
  select(-c(id, Company, Item)) %>%
  pivot_longer(-k_cluster, names_to = "attribute")
```

visualize cluster summary metrics information 
```{r}
df_clr_data %>%
  ggplot(aes(x = k_cluster, y = value)) + 
  facet_wrap(~attribute, scales = "free_y") + 
  geom_boxplot() + 
  labs(title = "Attribute Summary by Cluster", 
       x = "Cluster Number", 
       y = "") + 
  theme(title = element_text(size = 16, face = "bold"), 
        axis.title = element_text(size = 14, face = "bold"), 
        axis.text = element_text(size = 12), 
        legend.title = element_text(size = 10))
