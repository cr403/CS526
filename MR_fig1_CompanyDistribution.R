# Distributions by Company 
```{r}
# Bar chart
df %>%
  group_by(Company) %>%
  summarise(count = n()) %>%
  ggplot(aes(x = reorder(Company, -count), y = count, fill = Company)) + 
  geom_bar(stat = "identity") + 
  geom_text(aes(label = count), vjust = -0.5, size = 5 ) +
  scale_fill_manual(values = customPalette, name = "Company") + 
  labs(title = "Distribution of Menu Items Across Companies", 
       y = "Number of menu items", 
       x = "Company") 

# Pie chart
df %>%
  group_by(Company) %>%
  summarise(count = n(), .groups = "drop") %>%
  mutate(total = sum(count),
         percent = (count/total) *100) %>%
  ggplot(aes(x = "", y = count, fill = Company)) + 
  geom_bar(stat = "identity", width = 1)  + 
  coord_polar(theta = "y") + 
  geom_text(aes(label = paste(round(percent, 1), "%")), 
            color = "white", 
            fontface = "bold", 
            position = position_stack(vjust = .5)) + 
  scale_fill_manual(values = customPalette, name = "Company") + 
  labs(title = "Distribution of Menu Items Across Companies", 
       y = "Number of menu items", 
       x = "Company") +
  theme_void() 
```
