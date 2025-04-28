df_pdv %>%
  pivot_longer(cols = pdv_cal:pdv_cal_fat, names_to = "attribute", values_to = "percent_dv") %>%
  ggplot(aes(x = k_cluster, y = percent_dv)) + 
  facet_wrap(~attribute, scales = "free_y") + 
  geom_boxplot() + 
  labs(title = "Attribute Summary by Cluster", 
       x = "Cluster Number", 
       y = "%DV") + 
  theme(title = element_text(size = 16, face = "bold"), 
        axis.title = element_text(size = 14, face = "bold"), 
        axis.text = element_text(size = 12), 
        legend.title = element_text(size = 10))
