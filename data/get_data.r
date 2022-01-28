library(tidyverse)
library(fredr)

# pass your api key as a command line argument
args <- commandArgs(trailingOnly=TRUE)
fredr_set_key(args[1])
meta_data <- read_csv("meta_data.csv")

data <- data.frame(
  date = seq(as.Date("1900-01-01"), as.Date(paste0(substr(Sys.Date(), 1, 4), "-", substr(Sys.Date(), 6, 7), "-01")), by="1 month")
)

for (col in toupper(meta_data$series)) {
  tmp <- fredr(col)
  tmp <- tmp %>%
    rename(!!sym(tolower(col)) := value) %>%
    select(date, tolower(col)) %>%
    mutate(date = as.Date(date))
  data <- data %>%
    left_join(tmp, by="date") %>%
    mutate(!!sym(tolower(col)) := ifelse(!!sym(tolower(col)) == ".", NA, !!sym(tolower(col)))) %>%
    tibble()
}

first_gdp <- data %>% 
  filter(!is.na(gdpc1)) %>%
  select(date) %>%
  slice(1:1) %>%
  pull()
# only keep data after GDP started being published
data <- data %>%
  filter(date >= first_gdp) %>%
  arrange(desc(date))

# move quarterly values to last month of quarter instead of first
for (i in 1:nrow(meta_data)) {
  freq <- meta_data[i, "freq"] %>% pull()
  col <- meta_data[i, "series"] %>% pull()
  
  if (freq == "q") {
    data[,col] <- lead(data[,col], 2)
  }
}


write_csv(data %>% arrange(desc(date)), "data_raw.csv") # newest first so columns propoerly parsed in read_csv