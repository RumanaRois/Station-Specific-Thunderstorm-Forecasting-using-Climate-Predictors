# load_station_data.R
#
# Helper function shared by the per-station scripts (Sylhet_TS_R.R,
# Sreemangal_TS_R.R, Mymensingh_TS_R.R, All_Stations_Combine_Plot.R).
#
# S1_Dataset.xlsx stores all three stations side by side in one sheet
# ("Data"), as three repeated 9-column blocks:
#   Station | Year | Month | Monthly_TSF | Avg_Temp | Avg_RH |
#   Avg_CloudC | Avg_Rainfall | Avg_APressure
#
# This function extracts one station's block and renames the columns
# to match what the rest of the analysis scripts expect (Y, M, MT, ...).

library(readxl)

load_station_data <- function(path, station_name, sheet = "Data") {

  raw <- read_excel(path, sheet = sheet)

  col_names <- c("Station", "Y", "M", "MT", "Avg_Temp", "Avg_RH",
                 "Avg_CloudC", "Avg_Rainfall", "Avg_APressure")

  block_starts <- c(1, 10, 19)  # Sylhet, Sreemangal, Mymensingh

  station_blocks <- lapply(block_starts, function(start) {
    block <- as.data.frame(raw[, start:(start + 8)])
    names(block) <- col_names
    block
  })

  all_data <- do.call(rbind, station_blocks)

  D1 <- all_data[all_data$Station == station_name, ]
  D1 <- D1[order(D1$Y, D1$M), ]
  rownames(D1) <- NULL

  # Kept for backward-compatibility with scripts that reference $Year
  D1$Year <- D1$Y

  return(D1)
}
