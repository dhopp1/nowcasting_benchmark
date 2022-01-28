library(tidyverse)

# growth rates
growth_rate <- function (data) {
  for (col in colnames(data)[2:length(colnames(data))]) {
    # extracting actual series for growth rate
    series <- data %>%
      select(date, col) %>%
      drop_na(col)
   
    series <- series %>%
      mutate(!!sym(col) := !!sym(col) / lag(!!sym(col)) - 1)
    
    data <- data %>%
      select(-col) %>%
      left_join(series, by="date")
  }
  return (data)
}

data <- read_csv("data_raw.csv") %>%
  arrange(date) %>%
  growth_rate()

write_csv(data, "data_tf.csv")