####################################################################################
# Date: Tuesday, April 28, 2020
# Purpose: Scrape NBA Data

setwd("/Users/petertea/Documents/Sports-Analytics/NBA/")

# Use the bballR package to gather some end of season boxscore stats
# install.packages("remotes")
# remotes::install_github("bobbyingram/bballR")

# --> Load all libraries we will need
library(bballR)


# --> First scrape key data outlining info on all NBA players (this may help with merging datasets later on)
players_dat <- bballR::scrape_all_players()

# --> Save file
saveRDS(players_dat, file = "./per_100_poss/players_dat.RDS")
# Data Key: https://rdrr.io/github/bobbyingram/bballR/man/scrape_all_players.html


# --> Second, scrape end of season boxscore statistics (per 100 possessions) from the 1989 season to the previous 2019 NBA season.

for (year in 1989:2019){
  filename = paste("./per_100_poss/NBA_", year, "_P100P.RDS", sep = "" )
  saveRDS(bballR::scrape_season_per_100_poss(year), file = filename)
}
#https://rdrr.io/github/bobbyingram/bballR/man/scrape_season_per_100_poss.html


# --> Combine all seasons into a single "master file"
# --> Only include players who played more than 100 minutes a season
Historical_dat_1989_2019 <- data.frame()

for(year in 1989:2019){
  dat <- readRDS(paste("./per_100_poss/NBA_", year, "_P100P.RDS", sep = "" ))
  
  #dat <- dat %>%
  #  filter(MP > 100) %>% 
  #  filter(Tm != "TOT") %>%
  #  mutate(Year = year)
  
  Historical_dat_1989_2019 <- rbind(Historical_dat_1989_2019, dat)
  
}

write.csv(file = "./per_100_poss/per100poss_historic_data.csv",
          x = Historical_dat_1989_2019, row.names = FALSE)


##########################################################################################
# ----- Get list of player names who made the ALL - NBA Teams ----- #
library(rvest)
library(dplyr)

my_url <- read_html("https://www.basketball-reference.com/awards/all_league.html")
node <- "tbody .left , .right"
# Node found using InspectorGadget chrome tool.

scraped_data <- my_url %>%
  html_nodes(node) %>%
  html_text()

dummy<- scraped_data[1:984]
# Anything past the 984th element of the list is not required

# Remove empty elements from the list
to_remove <- c()
for (i in seq(from=25, to = 953, by = 32)){
  ind <- seq(from = i, to = i+7, by = 1)
  to_remove <- c(to_remove, ind)
}

dummy <- dummy[-to_remove]

# Remove entries that indicate type of All-NBA team awarded
to_remove2 <- which(grepl("0|1|2|3|NBA", dummy))
players <- dummy[-to_remove2]


# Initialize data frame...
year <- rep(2019:1989, each = 15)
team <- rep(rep(1:3, each = 5), times = 31)


ALL_NBA_data <- data.frame(Year = year, Team = team, Player = players, stringsAsFactors = F) %>%
  mutate(Position = substr(Player, start = nchar(Player), stop = nchar(Player)),
         Player = substr(Player, 1, nchar(Player) - 2)
  ) 


write.csv(file = "./per_100_poss/historic_all_nba_selections.csv",
          x = ALL_NBA_data, row.names = FALSE)

###########################################################################################
##### ----- Get current per 100 possession season stats ----- #####
my_url <- read_html("https://www.basketball-reference.com/leagues/NBA_2020_per_poss.html")

node <- ".left , .right , .center"


scraped_data <- my_url %>%
  html_nodes(node) %>%
  html_text()

# Rk rows (rows with column names) repeats 
Rk_ind <- which(scraped_data == "Rk")[-1]

garbage = vector()
for(index in Rk_ind){
  garbage=c(garbage, index:(index+31) )
  
} 
scraped_data <- scraped_data[-garbage]
  


each_row <- seq(from = 33, to = length(scraped_data) - 32,  by = 32)
col_names = scraped_data[1:32]

dat = matrix(ncol = 32)
for (i in each_row){
  dat <- rbind(dat, scraped_data[i:(i+31)])
  
}

colnames(dat) <- col_names

Current_season_dat <- as.data.frame(dat, stringsAsFactors = FALSE)
#Current_season_dat <- Current_season_dat[-which(Current_season_dat$Rk == "Rk"), ]


Current_season_dat[c(4, 6:32)] <- sapply(Current_season_dat[c(4, 6:32)], as.numeric)

Current_season_dat <- Current_season_dat[-1, c(-1,-30)]

write.csv(file = "./per_100_poss/2020_nba_season_data.csv",
          x = Current_season_dat, row.names = FALSE)


######################################################################################
# Scrape Historic Team wins data

my_url <- read_html("https://www.basketball-reference.com/leagues/NBA_wins.html")

node <- "#active_franchises tbody .right , #active_franchises a , #active_franchises .center"

scraped_data <- my_url %>%
  html_nodes(node) %>%
  html_text()

end_index = which(scraped_data =="1987-88")

scraped_data <- scraped_data[-((end_index-1):length(scraped_data))]

# Rk rows (rows with column names) repeats 
Rk_ind <- which(scraped_data == "Rk")[-1]

garbage = vector()
for(index in Rk_ind){
  garbage=c(garbage, index:(index+32) )
  
} 

scraped_data <- scraped_data[-garbage]

each_row <- seq(from = 34, to = length(scraped_data) - 31,  by = 33)
col_names = scraped_data[1:33]

dat = matrix(ncol = 33)

for (i in each_row){
  dat <- rbind(dat, scraped_data[i:(i+32)])
  
}

colnames(dat) <- col_names




historic_team_wins <- as.data.frame(dat, stringsAsFactors = FALSE)
#Current_season_dat <- Current_season_dat[-which(Current_season_dat$Rk == "Rk"), ]

historic_team_wins <- historic_team_wins[-1, c(-1,-3)]


write.csv(file = "./per_100_poss/historic_team_wins.csv",
          x = historic_team_wins, row.names = FALSE)





