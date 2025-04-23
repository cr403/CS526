# Plot percentage of missing attributes by company 
```{r}
missing_data <- df %>%
  mutate(Company = as.factor(Company)) %>%
  gather(key = "Variable", value = "Value", -Company) %>%
  group_by(Company, Variable) %>%
  summarise(Missing_Percentage = sum(is.na(Value)) / n() * 100, .groups = "drop")

ggplot(missing_data, aes(x = reorder(Variable, -Missing_Percentage), y = Missing_Percentage, fill = Company)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ Company) +
  coord_flip() +
  scale_fill_manual(values = customPalette) + 
  labs(title = "Percentage of Missing Feature Values by Company",
       x = "",
       y = "Percent of Menu Items") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
```
