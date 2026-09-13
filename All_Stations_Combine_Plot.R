setwd("D:/Mahin and Amrin Papers/Paper Mam/2. Clean Data")
create_station_plots <- function(file, station_name){
  
  library(ggplot2)
  library(dplyr)
  library(lubridate)
  library(viridis)
  library(forecast)
  library(seasonal)
  library(tidyr)
  
  # Load data
  df <- read.csv(file)
  
  # Preprocess
  df <- df %>%
    mutate(
      Year = as.integer(Y),
      Month = as.integer(M),
      Day = 1,
      Date = as.Date(paste(Year, Month, Day, sep="-")),
      month_num = Month,
      month_label = factor(month.abb[month_num], levels = month.abb)
    )
  
  # Time series object
  ts1 <- ts(df$MT, frequency = 12, start = c(min(df$Year),1))
  
  # STL
  STL <- stl(ts1, s.window = 5)
  df$Seasonally_Adjusted <- as.numeric(seasadj(STL))
  df$Trend <- as.numeric(trendcycle(STL))
  
  # Long format
  data_long <- df |> 
    pivot_longer(
      cols = c(MT, Seasonally_Adjusted, Trend),
      names_to = "Component",
      values_to = "Value"
    )
  
  # ---- Plot 1 (Time series) ----
  p1 <- ggplot(data_long, aes(Date, Value, color=Component)) +
    geom_line(linewidth=1) +
    scale_color_manual(values=c(
      "MT"="#50C878",
      "Seasonally_Adjusted"="red",
      "Trend"="steelblue"),
      labels = c(
        "MT" = "Monthly TS Frequency",
        "Seasonally_Adjusted" = "Seasonal-Adjusted",
        "Trend" = "Trend"
      )) +
    labs(
      title = station_name,
      x = "Year",
      y = paste("Monthly TS Frequency in", station_name)
    ) +
    theme_minimal() +
    theme(
      legend.position = "top",
      legend.title = element_blank(),
      panel.border = element_rect(color="black", fill=NA),
      plot.title = element_text(
        face = "bold",
        size = 12,
        hjust = 0   # left aligned
      )
    )
  
  # ---- Plot 2 (Seasonal plot) ----
  month_text <- data.frame(
    x=c(1.2,2:11,11.8),
    y=max(df$MT,na.rm=TRUE)+1,
    label=month.abb
  )
  
  p2 <- ggplot(df, aes(month_num, MT, group=Year, color=Year)) +
    geom_line(alpha=0.6) +
    scale_color_viridis_c(option="plasma", direction=-1) +
    scale_x_continuous(
      limits=c(1,12),
      breaks=1:12,
      labels=NULL,
      expand=expansion(add=0.3)
    ) +
    labs(
      x="Season",
      y=paste("Monthly TS Frequency in", station_name)
    ) +
    theme_minimal() +
    theme(
      panel.border=element_rect(color="black", fill=NA),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      legend.position = "top"
    ) +
    geom_vline(xintercept=seq(1.5,11.5,1), linetype="dashed", color="grey70") +
    geom_text(
      data=month_text,
      aes(x=x,y=y,label=label),
      inherit.aes=FALSE,
      color="blue"
    )
  
  # ---- Plot 3 (Boxplot) ----
  df$M <- factor(df$M, labels=month.abb)
  library(RColorBrewer)
  
  p3 <- ggplot(df, aes(M, MT, fill=M)) +
    geom_boxplot(color="black") +
    scale_fill_manual(values = brewer.pal(12, "Paired")) +
    labs(
      x="Month",
      y=paste("Monthly TS Frequency in", station_name)
    ) +
    theme_minimal() +
    theme(
      legend.position="none",
      panel.border=element_rect(color="black", fill=NA)
    )
  
  return(list(p1,p2,p3))
}

sylhet <- create_station_plots("Sylhet_MTS_Data.csv","Sylhet")
sreemangal <- create_station_plots("Sreemangal_MTS_Data.csv","Sreemangal")
mymensingh <- create_station_plots("Mymensingh_MTS_Data.csv","Mymensingh")

library(patchwork)

final_plot =
  (sylhet[[1]] + sylhet[[2]] + sylhet[[3]]) /
  (sreemangal[[1]] + sreemangal[[2]] + sreemangal[[3]]) /
  (mymensingh[[1]] + mymensingh[[2]] + mymensingh[[3]])

final_plot

ggsave(
  "Figure1_AllStations.tiff",
  final_plot,
  width = 14,
  height = 12,
  dpi = 300
)
