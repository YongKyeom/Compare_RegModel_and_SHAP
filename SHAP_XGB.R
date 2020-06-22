
####################################################################################
## XGBoost를 활용해 로딩량에 영향을 끼치는 인자들의 점도수준별 영향도 파악(by SHAP)
####################################################################################

#### 0. Set Options ####
rm(list = ls())
gc()

options(scipen = 100, max.print = 5000, stringsAsFactors = FALSE,
        repos = c(CRAN = "http://cran.rstudio.com"))

DEFALUT_FOLD <- ".."
setwd(DEFALUT_FOLD)

#### 1. library load ####
package <- c("data.table", "tidyverse", "DescTools", "lubridate", "scales", "ggrepel", "gridExtra", "RJDBC",
             "caret", "h2o", "glmnet", "rpart", "randomForest", "xgboost", "SHAPforxgboost",
             "foreach", "parallel", "doParallel", "doSNOW")
sapply(package, require, character.only = TRUE)

filter     <- dplyr::filter
lag        <- dplyr::lag
wday       <- lubridate::wday
month      <- lubridate::month
week       <- data.table::week
between    <- dplyr::between
row_number <- dplyr::row_number


## Set time zone
Sys.setenv(TZ = "Asia/Seoul")
Sys.timezone(location = TRUE)

## 시스템 사양
NCmisc::top(CPU = FALSE, RAM = TRUE)
NCmisc::memory.summary(unit = "mb")
parallel::detectCores()

TODAY <- Sys.Date()

#####################
#### Start ROOF! ####
##################### 

ST_TIME      <- Sys.time()

#### .. 저장경로 ####
SAVE_FOLD <- sprintf("03.result/SHAP/%s", gsub("-", "", TODAY))
if( !dir.exists(SAVE_FOLD) ) { dir.create(SAVE_FOLD, recursive = TRUE) }

#### 2. Data Load ####
analData    <- fread("01.data/dataset.csv")

#### .. Modify Column's name ####
COL_NAMES <- colnames(analData)
COL_NAMES <- gsub("\\(", "_", COL_NAMES)
COL_NAMES <- gsub("\\)", "", COL_NAMES)
COL_NAMES <- gsub("/", "_", COL_NAMES)
COL_NAMES <- gsub("#", "", COL_NAMES)
COL_NAMES <- gsub("\\.", "", COL_NAMES)
COL_NAMES <- gsub("=", "", COL_NAMES)
COL_NAMES <- gsub("-", "", COL_NAMES)

colnames(analData) <- COL_NAMES

VAR_LS     <- colnames(analData)[6:79]
TARGET_VAR <- "LOADMEAN"

#### .. Filter Needed columns ####
analData <- analData %>% 
  .[, TIME := lubridate::ymd_hms(TIME)] %>% 
  dplyr::select_(.dots = c("LOT_ID", "DATE", "TIME", VAR_LS, TARGET_VAR)) %>% 
  .[order(TIME)] %>% 
  setnames(c("LOADMEAN"), "LOAD_MEAN")

TARGET_VAR <- "LOAD_MEAN"


#### 3. NA Column Check ####
sapply(analData, function(x) sum(is.na(x)))

#### 4. Define User's Function & Variable ####
fnShowMemoryUse <- function(sort = "size", decreasing = TRUE, limit = 10) {
  
  objectList <- ls(parent.frame())
  
  oneKB <- 1024
  oneMB <- 1024 * 1024
  oneGB <- 1024 * 1024 * 1024
  
  memoryUse <- sapply(objectList, function(x) as.numeric(object.size(eval(parse(text = x)))))
  
  memListing <- sapply(memoryUse, function(size) {
    if (size >= oneGB)      return(paste(round(size/oneGB, 2), "GB"))
    else if (size >= oneMB) return(paste(round(size/oneMB, 2), "MB"))
    else if (size >= oneKB) return(paste(round(size/oneKB, 2), "KB"))
    else return(paste(size, "Bytes"))
  })
  
  memListing <- data.frame(objectName = names(memListing),
                           memorySize = memListing,
                           row.names  = NULL)
  
  if (sort == "alphabetical") {
    memListing <- memListing[order(memListing$objectName, decreasing = decreasing),]
  } else {
    memListing <- memListing[order(memoryUse, decreasing = decreasing),] 
    # will run if sort not specified or "size"
  }
  
  if(!is.na(limit)) { memListing <- memListing[1:limit,] }
  
  print(memListing, row.names = FALSE)
  return(invisible(memListing))
}
fnShowMemoryUse()


fnScatterPlot <- function(dataTable, X_VAR, Y_VAR, GROUP = NULL, METHOD = "lm", TITLE = NULL){
  
  # dataTable <- cssData
  # X_VAR <- "X160L.TANK.RPM"
  # Y_VAR <- "Viscosity.1"
  # GROUP <- NULL
  # METHOD <- "lm"
  
  if( is.null(GROUP) ){
    tmpPlot <- dataTable %>% 
      ggplot(aes_string(x = X_VAR, y = Y_VAR)) +
      geom_point() +
      geom_smooth(method = METHOD, se = TRUE)
  } else {
    tmpPlot <- dataTable %>% 
      ggplot(aes_string(x = X_VAR, y = Y_VAR, fill = GROUP, col = GROUP)) +
      geom_point() +
      geom_smooth(method = METHOD, se = TRUE)
  }
  
  if( !is.null(TITLE) ){
    tmpPlot <- tmpPlot +
      labs(title = TITLE) +
      theme(legend.position = "bottom",
            plot.title = element_text(hjust = 0.5, size = 15))
  }
  
  return(tmpPlot)
  
}

#######################################################################################

#### 5. Set Train / Valid  ####
trainData <- analData[DATE < "2020-01-02"]
validData <- analData[DATE == "2020-01-02"]
testData  <- analData[DATE >  "2020-01-02"]

# TRAIN_IDX <- createDataPartition(analData$DATE, p = 0.8)$Resample1
# trainData <- analData[TRAIN_IDX]
# testData  <- analData[-TRAIN_IDX]


# #### .. Preprocess of Variables ####
# preProc <- preProcess(trainData %>% dplyr::select_(.dots = VAR_LS),
#                       method      = c("center", "scale"),
#                       rangeBounds = c(0, 1))
# 
# trainData <- predict(preProc, trainData)
# testData  <- predict(preProc, testData)


#### 6. Fit Modeling ####
#### ..6.1 Xgbost ####
#### .... Grid Seach for Parameter ####
NTREE <- 5000

PARAGRID_LS <- expand.grid(subsample        = c(0.7), 
                           colsample_bytree = c(0.5, 0.7),
                           max_depth        = c(4, 8, 12, 16),
                           min_child        = 1, 
                           eta              = c(0.01, 0.05, 0.1)
)

#### .... Execute Grid Search ####
if( NROW(PARAGRID_LS) > 1 ){
  
  CORE_CNT     <- min(3, parallel::detectCores() - 1)
  MAIN_CLUSTER <- makeCluster(CORE_CNT)
  registerDoSNOW(MAIN_CLUSTER)
  
  CORE_SUB_CNT <- 6
  
  # PB       <- txtProgressBar(max = NROW(FILE_LS), style = 3)
  # PROGRESS <- function(n) { setTxtProgressBar(PB, n) }
  
  IterName       <- apply(PARAGRID_LS, 1, function(x) 
    sprintf("Subsample:%s / Colsample:%s / MaxDepth:%s / eta:%s", x[1], x[2], x[3], x[5])
  )
  
  RemainIterName <- IterName
  EndIterCnt     <- 0
  TotalIterCNT   <- NROW(IterName)
  ST_TIME        <- Sys.time()
  
  PROGRESS       <- function(n, x){
    
    RemainIterName <<- setdiff(RemainIterName, IterName[x])
    
    cat(sprintf("==== %s / %s(%s) is End with %s Mins! ====", 
                n, TotalIterCNT, IterName[x], 
                round(as.numeric(difftime(Sys.time(), ST_TIME, units = "mins")), 1)), 
        "\n")
    
    # if ( (TotalIterCNT - n) <= 5 & (TotalIterCNT - n) >= 1 ){
    #   cat(sprintf("==== Remain Process : %s ====", paste(RemainIterName, collapse = ", ")), "\n", "\n")
    # }
    
    if ( TotalIterCNT == n ){
      cat(sprintf("==== CV Process is End! %s ====", Sys.time()), "\n", "\n")
    }
  }
  ######################################################################
  
  OPTS     <- list(progress = PROGRESS)
  
  paraResult <- foreach(i0 = 1:NROW(PARAGRID_LS)
                        , .combine = function(x, y) { rbindlist(list(x, y), fill = TRUE) }
                        , .errorhandling = "stop"
                        , .options.snow  = OPTS
                        , .packages      = c("data.table", "tidyverse", "lubridate", "scales", 
                                             "caret", "xgboost", "SHAPforxgboost",
                                             "foreach", "parallel", "doParallel", "doSNOW")
  ) %dopar% {
    
    # i0 <- 1
    
    tmpSubsample <- PARAGRID_LS[i0, ]$subsample
    tmpColsample <- PARAGRID_LS[i0, ]$colsample_bytree
    tmpMaxdepth  <- PARAGRID_LS[i0, ]$max_depth
    tmpMinchild  <- PARAGRID_LS[i0, ]$min_child
    tmpEta       <- PARAGRID_LS[i0, ]$eta
    
    cat(sprintf("%s/%s \n %s \n at %s",
                i0, NROW(PARAGRID_LS),
                sprintf("Subsample:%s / Colsample:%s / Maxdepth:%s / \n Minchild:%s / Eta:%s", 
                        tmpSubsample, tmpColsample, tmpMaxdepth, tmpMinchild, tmpEta), 
                Sys.time()), 
        "\n",
        "\n")
    
    tmpXgboost <- xgb.train(
      data                  = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                          label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
      seed                  = 2020,
      nrounds               = NTREE,
      subsample             = tmpSubsample,
      colsample_bytree      = tmpColsample,
      max_depth             = tmpMaxdepth,
      min_child_weight      = tmpMinchild,
      eta                   = tmpEta,
      nthread               = CORE_SUB_CNT,
      early_stopping_rounds = 15,
      verbose               = 1,
      metrics               = "rmse"
      , watchlist             = list(
        train = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                            label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
        test = xgb.DMatrix(data  = data.matrix(validData %>% dplyr::select_(.dots = VAR_LS)),
                           label = validData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
      )
    )
    
    
    # ## Train vs Valid's RMSE Plot
    # tmpXboost_RMSE_Log <- tmpXgboost$evaluation_log %>%
    #   dplyr::select(iter, train_rmse, test_rmse) %>%
    #   reshape2::melt(id.vars = "iter") %>%
    #   rename(Iter = iter, Category = variable, RMSE = value)
    # 
    # tmpXboost_RMSE_Log %>%
    #   # dplyr::filter(RMSE <= 1) %>%
    #   ggplot() +
    #   geom_point(aes(x = Iter, y = RMSE, group = Category, col = Category),
    #              alpha = 0.5, size = 1)
    
    BEST_ITER  <- tmpXgboost$best_iteration
    ## Train
    TRAIN_RMSE <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$train_rmse
    ## Valid
    VALID_RMSE <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$test_rmse
    
    tmpRMSE_Result <- data.table(
      SUBSAMPLE         = tmpSubsample,
      COLSAMPLE_BYTREE  = tmpColsample,
      MAX_DEPTH         = tmpMaxdepth,
      MIN_CHILD_WEIGHT  = tmpMinchild,
      ETA               = tmpEta,
      BEST_ITER         = BEST_ITER,
      TRAIN_RMSE        = TRAIN_RMSE,
      VALID_RMSE        = VALID_RMSE
    )
    
    return(tmpRMSE_Result)
  }
  
  stopCluster(MAIN_CLUSTER)
  
  paraResult <- paraResult %>% dplyr::arrange(VALID_RMSE)
  
  bestTunePara <- paraResult %>% 
    dplyr::filter(VALID_RMSE == min(VALID_RMSE)) %>% 
    dplyr::arrange(VALID_RMSE) %>% .[1, ]
  
  SUBSAMPLE         <- bestTunePara$SUBSAMPLE
  COLSAMPLE_BYTREE  <- bestTunePara$COLSAMPLE_BYTREE
  MAX_DEPTH         <- bestTunePara$MAX_DEPTH
  MIN_CHILD_WEIGHT  <- bestTunePara$MIN_CHILD_WEIGHT
  ETA               <- bestTunePara$ETA
  BEST_ITER         <- round(bestTunePara$BEST_ITER, -2)
  
  
  FitXgboost <- xgboost(
    data                  = data.matrix(rbindlist(list(trainData, validData)) %>% dplyr::select_(.dots = VAR_LS)),
    label                 = (rbindlist(list(trainData, validData)) %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()),
    seed                  = 2020,
    nround                = BEST_ITER,
    subsample             = SUBSAMPLE,
    colsample_bytree      = COLSAMPLE_BYTREE,
    max_depth             = MAX_DEPTH,
    min_child_weight      = MIN_CHILD_WEIGHT,
    eta                   = ETA,
    nthread               = CORE_SUB_CNT,
    early_stopping_rounds = 5,
    verbose               = 0
  )
  
} else {
  
  ## Do not Grid Search 
  FitXgboost <- xgboost(
    data                  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
    label                 = (trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()),
    seed                  = 2020,
    nround                = NTREE,
    subsample             = PARAGRID_LS$subsample,
    colsample_bytree      = PARAGRID_LS$colsample_bytree,
    max_depth             = PARAGRID_LS$max_depth,
    min_child_weight      = PARAGRID_LS$min_child,
    eta                   = PARAGRID_LS$eta,
    nthread               = CORE_SUB_CNT,
    early_stopping_rounds = 5,
    verbose               = 0
  )
  
}
########################################################

#### .... Variable Importance ####
xgb.importance(model = FitXgboost)
xgb.plot.importance(xgb.importance(model = FitXgboost), top_n = 10)





#### 7. SHAP Algorithm ####
## To return the SHAP values and ranked features by mean|SHAP|
shap_Values <- shap.values(xgb_model = FitXgboost,
                           X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))
## The ranked features by mean |SHAP|
shap_Values$mean_shap_score


## To prepare the long-format data:
shap_Long <- shap.prep(xgb_model = FitXgboost, 
                       X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))

#### .. SHAP summary plot ####
shap.plot.summary(shap_Long)


#### .. diluted points ####
shap.plot.summary(shap_Long, x_bound  = 1.2, dilute = 10)


#### .. Alternatives ways to make the same plot ####
#### .... Option 1: from the xgboost model ####
shap.plot.summary.wrap1(xgb_model = FitXgboost, 
                        X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))


#### .... Option 2: supply a self-made SHAP values dataset  ####
## (e.g. sometimes as output from cross-validation)
shap.plot.summary.wrap2(shap_Values$shap_score, 
                        as.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)))



#### .. Dependency Plot ####
# ## Prepare the data 
# shap_Int <- shap.prep.interaction(xgb_model = FitXgboost,
#                                   X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))


## SHAP Interaction effect plot 
depend_Plot_1 <- shap.plot.dependence(data_long = shap_Long,
                                      # data_int  = shap_Int,
                                      x = "TOP_MONO_PUMP", 
                                      y = "TOP_MONO_PUMP", 
                                      color_feature = "VISCOSITY_1",
                                      add_hist      = TRUE
)
grid.arrange(depend_Plot_1)

## Save
ggsave(file   = sprintf("%s/%s__%s__colorFeatur_%s__%s.png", 
                        SAVE_FOLD, 
                        "TOP_MONO_PUMP", "TOP_MONO_PUMP", "VISCOSITY_1",
                        gsub("-", "", TODAY)),
       plot   = grid.arrange(depend_Plot_1),
       width  = 45,
       height = 30,
       units  = "cm")






#### 8. Test Set's RMSE ####
#### .. Predict ####
Pred_Test <- predict(FitXgboost,
                     newdata = data.matrix(testData %>% dplyr::select_(.dots = VAR_LS)))

#### .. RMSE ####
RMSE_Test <- MLmetrics::RMSE(y_pred = Pred_Test,
                             y_true = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull())



Test_Compare_DF <- data.table(
  TIME = testData$TIME,
  REAL = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull(),
  PRED = Pred_Test
)


Test_Compare_DF %>% 
  ggplot() +
  geom_point(aes(x = TIME, y = REAL), col = "black") +
  geom_point(aes(x = TIME, y = PRED), col = "blue") +
  scale_x_datetime(labels = date_format("%Y-%m-%d %H:%M:%S")) +
  labs(title = "Test set's Observations vs Predicted Values") +
  theme(plot.title = element_text(hjust = 0.5, size = 15))


#### .. Obs vs Pred Plot ####
rbindlist(list(trainData, validData, testData), fill = TRUE) %>% 
  .[DATE >= "2020-01-01"] %>%
  # .[DATE >= "2020-01-03"] %>% 
  ggplot() +
  geom_point(aes(x = TIME, y = LOAD_MEAN)) +
  geom_vline(xintercept = lubridate::ymd_hms("2020-01-03 00:00:00"), 
             linetype = "dotted", color = "red") +
  scale_x_datetime(labels = date_format("%Y-%m-%d %H:%M:%S")) +
  geom_point(data = Test_Compare_DF,
             aes(x = TIME, y = PRED), col = "blue", size = 1, alpha = 0.1) +
  labs(title = "Total Observations vs Predicted Values") +
  theme(plot.title = element_text(hjust = 0.5, size = 15))







#### .. 2020-01-03's RMSE ####
MLmetrics::RMSE(
  y_pred = predict(FitXgboost,
                   newdata = data.matrix(
                     testData %>% 
                       dplyr::filter(DATE >  "2020-01-02") %>% 
                       dplyr::select_(.dots = VAR_LS)
                   )),
  y_true = testData %>%
    dplyr::filter(DATE >  "2020-01-02") %>% 
    dplyr::select_(.dots = TARGET_VAR) %>% pull()
)

Test_Compare_DF_Filter <- data.table(
  TIME = testData %>% dplyr::filter(DATE >  "2020-01-02") %>% .$TIME,
  REAL = testData %>% dplyr::filter(DATE >  "2020-01-02") %>% dplyr::select_(.dots = TARGET_VAR) %>% pull(),
  PRED = predict(FitXgboost,
                 newdata = data.matrix(
                   testData %>% 
                     dplyr::filter(DATE >  "2020-01-02") %>% 
                     dplyr::select_(.dots = VAR_LS)
                 ))
)


Test_Compare_DF_Filter %>% 
  ggplot() +
  geom_point(aes(x = TIME, y = REAL), col = "black") +
  geom_point(aes(x = TIME, y = PRED), col = "blue") +
  scale_x_datetime(labels = date_format("%Y-%m-%d %H:%M:%S")) +
  labs(title = "Test Set's Observations vs Predicted Values at 2020-01-03") +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

#### .. Obs vs Pred Plot ####
rbindlist(list(trainData, testData), fill = TRUE) %>% 
  .[DATE >= "2020-01-01"] %>%
  # .[DATE >= "2020-01-03"] %>%
  ggplot() +
  geom_point(aes(x = TIME, y = LOAD_MEAN)) +
  geom_vline(xintercept = lubridate::ymd_hms("2020-01-03 00:00:00"), 
             linetype = "dotted", color = "red") +
  scale_x_datetime(labels = date_format("%Y-%m-%d %H:%M:%S")) +
  geom_point(data = Test_Compare_DF_Filter,
             aes(x = TIME, y = PRED), col = "blue", size = 1, alpha = 1) +
  labs(title = "Total Observations vs Predicted Values") +
  theme(plot.title = element_text(hjust = 0.5, size = 15))





END_TIME <- Sys.time()

cat("\n",
    sprintf("===== Total Process is END! ====="),
    '\n Start : ', paste0(ST_TIME),
    '\n End   : ', paste0(END_TIME),
    '\n',
    capture.output(difftime(END_TIME, ST_TIME, units = "mins")),
    "\n",
    "\n")

