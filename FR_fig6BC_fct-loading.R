fact_load <- function(data, 
         pc) { 
  data %>%
  ggplot(aes(x = fct_reorder(attribute, .data[[pc]]), y = .data[[pc]])) + 
  geom_bar(stat = "identity") + 
  coord_flip() + 
  labs(title = paste0(pc, " Factor Loading"), 
       x = "") +
  theme(title = element_text(size = 16, face = "bold"), 
        axis.title = element_text(size = 14, face = "bold"), 
        axis.text = element_text(size = 12), 
        legend.title = element_text(size = 10))
}
rotation <- pca_result$rotation %>% data.frame() %>% rownames_to_column(var = "attribute") 
fact_load(rotation, "PC1")
fact_load(rotation, "PC2")
