df <- left_join(df_scores, df_clr_meta, by = c("Item", "Company", "Weight_Watchers_Pnts") 

df %>% 
  filter(Weight_Watchers_Pnts > 0) %>% 
  ggplot(aes(x = PC1, y = Weight_Watchers_Pnts)) + 
  geom_point(aes(color = k_cluster)) + 
  geom_smooth(stat = "smooth", method = lm, se = TRUE) + 
  stat_cor(label.x = -5, label.y = 20, color = "red") +
  labs(title = "PC1 x Weight_Watchers_Pnts") + 
  theme(title = element_text(size = 16, face = "bold"), 
        axis.title = element_text(size = 14, face = "bold"), 
        axis.text = element_text(size = 12), 
        legend.title = element_text(size = 10)) 
