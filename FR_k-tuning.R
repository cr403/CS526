library(caret) 
library(cluster) 
crossValidateKMeans <- function(data, 
                                k.values = 2:10, 
                                n.folds = 5, 
                                n.start = 10, 
                                seed = 123) {
  set.seed(seed)
  
  # Create folds (indices for training/validation)
  # This returns a list where each element is the training indices for that fold
  folds <- createFolds(1:nrow(data), k = n.folds, returnTrain = TRUE)
  
  results <- numeric(length(k.values))
  names(results) <- as.character(k.values)
  
  # Iterate over each candidate k
  for (k in k.values) {
    fold_silhouette_scores <- c()
    
    # Perform cross-validation
    for (i in seq_along(folds)) {
      train_indices <- folds[[i]]
      val_indices   <- setdiff(seq_len(nrow(data)), train_indices)
      
      train_data <- data[train_indices, , drop = FALSE]
      val_data   <- data[val_indices, , drop = FALSE]
      
      # Fit k-means on the training subset
      km_model <- kmeans(train_data, centers = k, nstart = n.start)
      
      # Predict cluster labels for the validation subset
      # (Here we assign each validation sample to the nearest cluster center)
      val_labels <- apply(val_data, 1, function(x) {
        # Compute squared Euclidean distance to each cluster center
        dist_to_centers <- apply(km_model$centers, 1, function(center) {
          sum((x - center)^2)
        })
        # Assign to nearest center
        which.min(dist_to_centers)
      })
      
      # Compute silhouette scores on the validation set
      # Requires a distance matrix of validation samples
      # If there's only one cluster, silhouette is undefined (NA)
      if (length(unique(val_labels)) > 1) {
        d_val <- dist(val_data)
        sil   <- silhouette(val_labels, d_val)
        fold_silhouette_scores <- c(fold_silhouette_scores, mean(sil[, 3]))
      } else {
        fold_silhouette_scores <- c(fold_silhouette_scores, NA)
      }
    }
    
    # Average silhouette score across folds for this k
    results[as.character(k)] <- mean(fold_silhouette_scores, na.rm = TRUE)
  }
  
  return(results)
}

cv_results <- crossValidateKMeans(data = df_clr)

data.frame(cv_results) %>% rownames_to_column(var = "k") %>% mutate(k = factor(k, levels = c("2", "3", "4", "5", "6", "7", "8", "9", "10"))) %>%
  ggplot(aes(x = k, y = cv_results, group = 1)) + 
  geom_point() + 
  geom_line() + 
  labs(title = "Tuning k by Cluster Stability", 
       x = "Number of clusters (k)", 
       y = "Mean Silhouette Score") + 
  theme(title = element_text(size = 14, face = "bold"), 
        axis.text = element_text(size = 12))
