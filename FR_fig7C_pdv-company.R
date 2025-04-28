df_pdv %>%
  pivot_longer(cols = pdv_cal:pdv_cal_fat, names_to = "attribute", values_to = "percent_dv") %>%
  filter(!attribute %in% good_att) %>%
  group_by(Company, Item) %>%
  summarise(exceeds_33 = any(percent_dv > 33), .groups = "drop") %>%
  group_by(Company) %>%
  summarise(percent_exceeding_33 = mean(exceeds_33) * 100) %>%
  ggplot(aes(x = Company, y = percent_exceeding_33, fill = Company)) + 
  geom_bar(stat = "identity") + 
  ylim(0, 100) + 
  scale_fill_manual(values = customPalette) + 
  labs(title = "%Items exceeding 33% DV", 
       y = "%Items") + 
  theme(title = element_text(size = 14, face = "bold"), 
        axis.text = element_text(size = 12), 
        legend.position = "none") 
