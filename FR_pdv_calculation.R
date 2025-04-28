```{r}
# Percent daily values based on average 2000 calorie diet 
df_pdv <- df_pca_k %>% 
  mutate(pdv_cal = Calories/2000*100, 
         pdv_tot_fat = Total_Fat_g/78*100, 
         Calories_from_Fat = Total_Fat_g * 9,
         pdv_sat_fat = Saturated_Fat_g/20*100,
         pdv_cho = Cholesterol_mg/300*100, 
         pdv_sod = Sodium_mg/2300*100,
         pdv_carb = Carbs_g/283*100, 
         pdv_fib = Fiber_g/28*100, 
         pdv_sug = Sugars_g/50*100,
         pdv_prot = Protein_g/50*100,
         pdv_cal_fat = Calories_from_Fat/600*100) %>%
  select(-c(Calories, Total_Fat_g, Trans_Fat_g, Calories_from_Fat, Saturated_Fat_g, Cholesterol_mg, Sodium_mg, Carbs_g, Fiber_g, Sugars_g, Protein_g))
```
