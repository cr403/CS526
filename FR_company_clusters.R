df_clr_meta %>%
  group_by(k_cluster, Company) %>%
  summarise(count = n()) %>%
  ggplot(aes(x = k_cluster, y = count, fill = Company)) + 
  geom_bar(stat= "identity") + 
  facet_wrap(~Company) + 
  scale_fill_manual(values = customPalette) + 
  labs(title = "Distribution of Clusters within Companies") + 
  theme(title = element_text(size = 14, face = "bold"), 
        axis.text = element_text(size = 12),
        strip.text = element_text(size =12)) 
