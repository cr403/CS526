```{r}
attCheck <- function(xVar = Company, # default 
                     yVar,
                     data) {
  data %>%
    ggplot(aes(x = {{xVar}}, y = {{yVar}}, fill = {{xVar}})) +
    geom_jitter(width = 0.2, alpha = 0.5, color = "skyblue") + 
    geom_boxplot(alpha = 0.5, outlier.shape = NA) + 
    scale_fill_manual(values = customPalette) + 
    labs(title = paste0("Distribution of ", deparse(substitute(yVar)), " by ", deparse(substitute(xVar))))
}
```

```{r}
attCheck(xVar = Company, yVar = Calories, data = df)
```
