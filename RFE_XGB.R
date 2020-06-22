
######################################################################################
## XGBoost를 활용해 로딩량에 영향을 끼치는 인자들을 선별(Recursive Feature Elimination)
######################################################################################

#### 0. Set Options ####
rm(list = ls())
gc()

options(scipen = 100, max.print = 5000, stringsAsFactors = FALSE,
        repos = c(CRAN = "http://cran.rstudio.com"))

DEFALUT_FOLD <- ".."
setwd(DEFALUT_FOLD)

#### 1. library load ####
package <- c("data.table", "tidyverse", "DescTools", "lubridate", "scales", 
             "ggrepel", "gridExtra", "RColorBrewer", "RJDBC",
             "caret", "xgboost", "progress",
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
SAVE_FOLD <- sprintf("%s/03.result/RFE_XGB/%s", DEFALUT_FOLD, gsub("-", "", TODAY))
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

VAR_LS           <- colnames(analData)[6:79]
TARGET_VAR       <- "LOADMEAN"

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
######################################################################
fnShowMemoryUse()



fnRFE_XGB <- function(dataTable,
                      VAR_LS,
                      TARGET_VAR,
                      SIZE_LS,
                      NTREE,
                      PARAGRID_LS  = NULL,
                      Fold_CNT     = 5,
                      Fold_Stand   = "DATE",
                      CORE_N       = 12,
                      Error_Method = "RMSE",
                      Verbose_TF   = TRUE){
  
  # dataTable <- trainData
  
  fnVerbose <- function(...){ if( isTRUE(Verbose_TF) ){ cat(...) } }
  ######################################################################
  
  tmpST_TIME <- Sys.time()
  SIZE_LS    <- SIZE_LS[order(SIZE_LS, decreasing = TRUE)]
  
  if( !is.null(Fold_Stand) ){
    ## Stratitied Sample
    indexList <- createFolds(dataTable %>% dplyr::select_(.dots = Fold_Stand) %>% pull(), 
                             k = Fold_CNT, 
                             returnTrain = TRUE)
  } else {
    ## Random Sample
    indexList <- createFolds(c(1:NROW(dataTable)), 
                             k = Fold_CNT, 
                             returnTrain = TRUE)
  }
  
  ## Roof is StarT!
  for( s0 in 1:NROW(SIZE_LS) ){
    
    # s0 <- 1
    
    tmpSIZE       <- SIZE_LS[s0]
    tmpRemove_CNT <- SIZE_LS[s0] - SIZE_LS[s0 + 1]
    
    if( s0 == 1 ){ 
      tmpVAR_LS     <- VAR_LS 
      tmpREMOVE_VAR <- NA
      
      ## Result DT
      CV_ResultSummary_Final <- NULL
      VAR_IMP_Summary_Final  <- NULL
    } else {
      tmpVAR_LS <- setdiff(tmpVAR_LS, tmpREMOVE_VAR)
    }
    
    fnVerbose(sprintf("==== Size : %s(%s/%s) at %s ====", tmpSIZE, s0, NROW(SIZE_LS), Sys.time()), "\n")
    
    CORE_CNT     <- min(2, parallel::detectCores() - 1)
    MAIN_CLUSTER <- makeCluster(CORE_CNT)
    registerDoSNOW(MAIN_CLUSTER)
    
    CORE_SUB_CNT <- floor(CORE_N / CORE_CNT)
    
    # PB       <- txtProgressBar(max = NROW(indexList), style = 3)
    # PROGRESS <- function(n) { setTxtProgressBar(PB, n) }
    # OPTS     <- list(progress = PROGRESS)
    
    fnListEachRbind <- function(...){ mapply("rbind", ..., SIMPLIFY = FALSE) }
    #####################################################################################
    
    CV_Result <- foreach(c1 = 1:NROW(indexList)
                         , .combine = fnListEachRbind
                         , .multicombine = TRUE
                         , .errorhandling = "stop"
                         # , .options.snow  = OPTS
                         , .packages      = c("data.table", "tidyverse", "lubridate", "scales", 
                                              "caret", "xgboost",
                                              "foreach", "parallel", "doParallel", "doSNOW")
    ) %dopar% {
      
      # c1 <- 1
      
      tmpIndex_LS <- indexList[[c1]]
      
      tmpXgboost <- xgb.train(
        data                  = xgb.DMatrix(
          data  = data.matrix(dataTable[tmpIndex_LS] %>% dplyr::select_(.dots = tmpVAR_LS)),
          label = dataTable[tmpIndex_LS] %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
        seed                  = 2020,
        nrounds               = NTREE,
        subsample             = 0.7,
        colsample_bytree      = 0.7,
        max_depth             = 6,
        min_child_weight      = 1,
        eta                   = 0.01,
        nthread               = CORE_SUB_CNT,
        early_stopping_rounds = 15,
        verbose               = 1,
        metrics               = tolower(Error_Method)
        , watchlist             = list(
          train = xgb.DMatrix(
            data  = data.matrix(dataTable[tmpIndex_LS] %>% dplyr::select_(.dots = tmpVAR_LS)),
            label = dataTable[tmpIndex_LS] %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
          test = xgb.DMatrix(
            data  = data.matrix(dataTable[-tmpIndex_LS] %>% dplyr::select_(.dots = tmpVAR_LS)),
            label = dataTable[-tmpIndex_LS] %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
        )
      )
      
      # ## Train vs Valid's RMSE Plot
      # tmpXboost_RMSE_Log <- tmpXgboost$evaluation_log %>%
      #   dplyr::select(iter, train_rmse, test_rmse) %>%
      #   reshape2::melt(id.vars = "iter") %>%
      #   rename(Iter = iter, Category = variable, RMSE = value)
      # 
      # tmpXboost_RMSE_Log %>%
      #   dplyr::filter(RMSE <= 5) %>%
      #   ggplot() +
      #   geom_point(aes(x = Iter, y = RMSE, group = Category, col = Category),
      #              alpha = 0.5, size = 1)
      
      tmpResult_Error <- data.table(
        ITER       = c1,
        TRAIN_RMSE = tmpXgboost$evaluation_log[iter == max(iter)]$train_rmse,
        TEST_RMSE  = tmpXgboost$evaluation_log[iter == max(iter)]$test_rmse,
        TRAIN_MAE  = MAE(
          pred = predict(tmpXgboost,
                         newdata = data.matrix(dataTable[tmpIndex_LS] %>% dplyr::select_(.dots = tmpVAR_LS))),
          obs  = dataTable[tmpIndex_LS] %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
        TEST_MAE   = MAE(
          pred = predict(tmpXgboost,
                         newdata = data.matrix(dataTable[-tmpIndex_LS] %>% dplyr::select_(.dots = tmpVAR_LS))),
          obs  = dataTable[-tmpIndex_LS] %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
      )
      
      tmpVarImp <- xgb.importance(model = tmpXgboost) %>% 
        dplyr::select(Feature, Gain) %>% 
        setnames(c("VAR_NAME", "VAR_IMPORT")) %>% 
        .[, VAR_IMPORT := VAR_IMPORT / max(VAR_IMPORT)]
      
      return(list(CV_RESULT = tmpResult_Error,
                  VAR_IMP   = tmpVarImp))
      
    }
    stopCluster(MAIN_CLUSTER)
    # close(PB)
    
    ## CV Result
    CV_ResultSummary <- CV_Result$CV_RESULT %>% 
      .[, list(TRAIN_RMSE_MEAN = mean(TRAIN_RMSE),
               TEST_RMSE_MEAN  = mean(TEST_RMSE),
               TRAIN_RMSE_SD   = sd(TRAIN_RMSE),
               TEST_RMSE_SD    = sd(TEST_RMSE),
               TRAIN_MAE_MEAN  = mean(TRAIN_MAE),
               TEST_MAE_MEAN   = mean(TEST_MAE),
               TRAIN_MAE_SD    = sd(TRAIN_MAE),
               TEST_MAE_SD     = sd(TEST_MAE))] %>% 
      .[, SIZE := tmpSIZE] %>% dplyr::select(SIZE, everything())
    CV_ResultSummary_Final <- rbind(CV_ResultSummary_Final, CV_ResultSummary)
    
    ## Variable Importance
    VAR_IMP_Summary <- CV_Result$VAR_IMP %>% 
      .[, list(VAR_IMPORT = mean(VAR_IMPORT)),
        by = c("VAR_NAME")] %>% 
      .[, VAR_IMPORT := VAR_IMPORT / max(VAR_IMPORT)] %>% 
      .[order(VAR_IMPORT, decreasing = TRUE)] %>% 
      .[, SIZE := tmpSIZE] %>% dplyr::select(SIZE, everything())
    VAR_IMP_Summary_Final <- rbind(VAR_IMP_Summary_Final, VAR_IMP_Summary)
    
    ## Remove Variable in Next Step
    tmpREMOVE_VAR <- try(VAR_IMP_Summary$VAR_NAME[NROW(VAR_IMP_Summary):(NROW(VAR_IMP_Summary) - tmpRemove_CNT + 1)],
                         silent = TRUE)
    
  } ## Iteration of Size(s0)
  
  ## Best Variable's CNT
  if( Error_Method == "RMSE" ){
    BestSize <- CV_ResultSummary_Final[TEST_RMSE_MEAN == min(TEST_RMSE_MEAN)]$SIZE
  } else {
    BestSize <- CV_ResultSummary_Final[TEST_MAE_MEAN == min(TEST_MAE_MEAN)]$SIZE
  }
  
  ## Best Variable remained
  BEST_VAR_LS <- VAR_IMP_Summary_Final %>% 
    .[SIZE == BestSize] %>% 
    .[order(VAR_IMPORT, decreasing = TRUE)] %>% .$VAR_NAME
  
  
  ## Fit with Best Variable
  if( is.null(PARAGRID_LS) ){
    ## Do not Grid Search and Set default Para 
    
    fnVerbose("\n",
              sprintf("==== With Size : %s, Fitting XGB with default Para at %s ====", BestSize, Sys.time()),
              "\n")
    
    Fit_XGB <- xgboost(
      data                  = data.matrix(dataTable %>% dplyr::select_(.dots = BEST_VAR_LS)),
      label                 = (dataTable %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()),
      seed                  = 2020,
      nround                = NTREE,
      subsample             = 0.5,
      colsample_bytree      = 0.5,
      eta                   = 0.01,
      nthread               = CORE_N,
      early_stopping_rounds = 15,
      verbose               = 0
    )
    
  } else {
    ## Input Parameter setting(s)
    
    if( NROW(PARAGRID_LS) > 1 ){
      ## Execute Grid Search
      
      fnVerbose("\n",
                sprintf("==== With Size : %s, Execute Grid Search at %s ====", BestSize, Sys.time()),
                "\n")
      
      tmpTrain <- dataTable[indexList[[1]]]
      tmpValid <- dataTable[-indexList[[1]]]
      
      CORE_CNT     <- min(2, parallel::detectCores() - 1)
      MAIN_CLUSTER <- makeCluster(CORE_CNT)
      registerDoSNOW(MAIN_CLUSTER)
      
      CORE_SUB_CNT <- floor(CORE_N / CORE_CNT)
      
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
                                                 "caret", "xgboost", 
                                                 "foreach", "parallel", "doParallel", "doSNOW")
      ) %dopar% {
        
        # i0 <- 1
        
        tmpSubsample <- PARAGRID_LS[i0, ]$subsample
        tmpColsample <- PARAGRID_LS[i0, ]$colsample_bytree
        tmpMaxdepth  <- PARAGRID_LS[i0, ]$max_depth
        tmpMinchild  <- PARAGRID_LS[i0, ]$min_child
        tmpEta       <- PARAGRID_LS[i0, ]$eta
        
        fnVerbose(sprintf("%s/%s \n %s \n at %s",
                          i0, NROW(PARAGRID_LS),
                          sprintf("Subsample:%s / Colsample:%s / Maxdepth:%s / \n Minchild:%s / Eta:%s", 
                                  tmpSubsample, tmpColsample, tmpMaxdepth, tmpMinchild, tmpEta), 
                          Sys.time()), 
                  "\n",
                  "\n")
        
        tmpXgboost <- xgb.train(
          data                  = xgb.DMatrix(
            data  = data.matrix(tmpTrain %>% dplyr::select_(.dots = BEST_VAR_LS)),
            label = tmpTrain %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
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
            train = xgb.DMatrix(data  = data.matrix(tmpTrain %>% dplyr::select_(.dots = BEST_VAR_LS)),
                                label = tmpTrain %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
            test = xgb.DMatrix(data  = data.matrix(tmpValid %>% dplyr::select_(.dots = BEST_VAR_LS)),
                               label = tmpValid %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
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
      
      ## Fit
      Fit_XGB <- xgboost(
        data                  = data.matrix(dataTable %>% dplyr::select_(.dots = BEST_VAR_LS)),
        label                 = (dataTable %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()),
        seed                  = 2020,
        nround                = max(BEST_ITER, 300),
        subsample             = SUBSAMPLE,
        colsample_bytree      = COLSAMPLE_BYTREE,
        max_depth             = MAX_DEPTH,
        min_child_weight      = MIN_CHILD_WEIGHT,
        eta                   = ETA,
        nthread               = CORE_N,
        early_stopping_rounds = 15,
        verbose               = 0
      )
      
    } else {
      ## Do not Grid Search 
      
      fnVerbose("\n",
                sprintf("==== With Size : %s, Fitting XGB with ParaList at %s ====", BestSize, Sys.time()),
                "\n")
      
      Fit_XGB <- xgboost(
        data                  = data.matrix(dataTable %>% dplyr::select_(.dots = BEST_VAR_LS)),
        label                 = (dataTable %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()),
        seed                  = 2020,
        nround                = max(NTREE, 300),
        subsample             = PARAGRID_LS$subsample,
        colsample_bytree      = PARAGRID_LS$colsample_bytree,
        max_depth             = PARAGRID_LS$max_depth,
        min_child_weight      = PARAGRID_LS$min_child,
        eta                   = PARAGRID_LS$eta,
        nthread               = CORE_N,
        early_stopping_rounds = 15,
        verbose               = 0
      )
      
    }
    
  }
  
  ## End!
  tmpEND_TIME <- Sys.time()
  
  fnVerbose("\n",
            sprintf("===== Total XGB's RFE Process is END! ====="),
            '\n Start : ', paste0(tmpST_TIME),
            '\n End   : ', paste0(tmpEND_TIME),
            '\n',
            capture.output(difftime(tmpEND_TIME, tmpST_TIME, units = "mins")),
            "\n",
            "\n")
  
  ## Output
  return(list(Final_Model            = Fit_XGB,                 # XGB Model
              Best_Size              = BestSize,                # Best Size of Variabls
              Best_Var_LS            = BEST_VAR_LS,             # Remained Variables
              VAR_IMP_Summary_Final  = VAR_IMP_Summary_Final,   # Variable Importances
              CV_ResultSummary_Final = CV_ResultSummary_Final)  # Cross Validation's Result
  )
  
}


#######################################################################################


#### 5. Set Train / Valid  ####
trainData <- analData[DATE <= "2020-01-02"]
testData  <- analData[DATE >  "2020-01-02"]

# TRAIN_IDX <- createDataPartition(analData$DATE, p = 0.8)$Resample1
# trainData <- analData[TRAIN_IDX]
# testData  <- analData[-TRAIN_IDX]


#### .. Preprocess of Variables ####
preProc <- preProcess(trainData %>% dplyr::select_(.dots = VAR_LS),
                      method      = c("corr"))

trainData <- predict(preProc, trainData)
testData  <- predict(preProc, testData)

VAR_LS <- setdiff(VAR_LS, preProc$method$remove)


#### 6. Fit RFE ####
#### ..6.2 XGB ####

MODEL     <- "HE2L"
CORE_N    <- 12
SIZE_LS   <- 4:NROW(VAR_LS)
NTREE     <- 700

PARAGRID_LS <- expand.grid(subsample        = c(0.7), 
                           colsample_bytree = c(0.5, 0.7),
                           max_depth        = c(4, 8, 12),
                           min_child        = 1, 
                           eta              = c(0.01, 0.05, 0.1)
)

#### .... Fit ####
Fit_RFE_XGB <- fnRFE_XGB(dataTable    = trainData,
                         VAR_LS       = VAR_LS,
                         TARGET_VAR   = TARGET_VAR,
                         SIZE_LS      = SIZE_LS,
                         NTREE        = NTREE,
                         PARAGRID_LS  = PARAGRID_LS,
                         Fold_CNT     = 5,
                         Fold_Stand   = "DATE",
                         CORE_N       = CORE_N,
                         Error_Method = "RMSE",
                         Verbose_TF   = TRUE)

#### .... Selected Variable ####
VAR_SELECTED_LS <- Fit_RFE_XGB$Best_Var_LS

#### .... Predict ####
Pred_RFE_XGB <- predict(Fit_RFE_XGB$Final_Model, 
                        newdata = data.matrix(testData %>% dplyr::select_(.dots = VAR_SELECTED_LS)))
RMSE_RFE_XGB <- RMSE(pred = Pred_RFE_XGB,
                     obs  = testData %>% select_(.dots = TARGET_VAR) %>% pull())
# RMSE_RFE

#### .... Variable's Importance ####
VariableImportance_RFE_XGB <- xgb.importance(model = Fit_RFE_XGB$Final_Model) %>% 
  dplyr::select(Feature, Gain) %>% 
  setnames(c("VAR_NAME", "VAR_IMPORT")) %>% 
  .[, VAR_IMPORT := VAR_IMPORT / max(VAR_IMPORT)] %>% 
  .[, MODEL  := MODEL] %>% 
  .[, TARGET := TARGET_VAR] %>% 
  .[, METHOD := "XGB"] %>% 
  dplyr::select(MODEL, TARGET, METHOD, VAR_NAME, VAR_IMPORT, everything())

write.csv(VariableImportance_RFE_XGB,
          file = sprintf("%s/%s_Var_RFE_Import_XGB_%s.csv", 
                         SAVE_FOLD, MODEL, gsub("-", "", Sys.Date())),
          row.names = FALSE,
          fileEncoding = "euc-kr")


#### .... Change of Variable Importance's ####
#### ..... by Rank ####
colorCNT   <- Fit_RFE_XGB$VAR_IMP_Summary_Final$VAR_NAME %>% unique %>% NROW
# display.brewer.all()

VariableImport_RankChange_Plot <- Fit_RFE_XGB$VAR_IMP_Summary_Final %>% 
  group_by(SIZE) %>% 
  dplyr::mutate(VAR_RANK = rank(-VAR_IMPORT, ties.method = "first")) %>% 
  ungroup %>% 
  # dplyr::filter(SIZE == 15) %>%
  ggplot() +
  geom_line(aes(x = SIZE,
                y = VAR_RANK,
                col = VAR_NAME)) +
  labs(title = sprintf("%s's %s \n Vairable수에 따른 변수중요도 순위 in XGB", MODEL, TARGET_VAR), 30,
       x = "CNT of Variables",
       y = "RANK") +
  theme(plot.title = element_text(hjust = 0.5, size = 15),
        legend.position = "bottom") +
  scale_color_manual(values = colorRampPalette(brewer.pal(8, "Accent"))(colorCNT))

## Save
ggsave(file   = sprintf("%s/%s_Var_Import_RankChange_Plot_RFE_XGB__%s.png", 
                        SAVE_FOLD, MODEL, gsub("-", "", TODAY)),
       plot   = VariableImport_RankChange_Plot,
       width  = 45,
       height = 30,
       units  = "cm")


#### ..... by Importance Point ####
VariableImport_PointChange_Plot <- Fit_RFE_XGB$VAR_IMP_Summary_Final %>% 
  ggplot() +
  geom_line(aes(x = SIZE,
                y = VAR_IMPORT,
                col = VAR_NAME)) +
  theme(legend.position = "bottom") +
  labs(title = sprintf("%s's %s \n Vairable수에 따른 변수중요도 점수 in XGB", MODEL, TARGET_VAR), 30,
       x = "CNT of Variables",
       y = "Point") +
  theme(plot.title = element_text(hjust = 0.5, size = 15),
        legend.position = "bottom") +
  scale_color_manual(values = colorRampPalette(brewer.pal(8, "Accent"))(colorCNT))

## Save
ggsave(file   = sprintf("%s/%s_Var_Import_PointChange_Plot_RFE_XGB__%s.png", 
                        SAVE_FOLD, MODEL, gsub("-", "", TODAY)),
       plot   = VariableImport_PointChange_Plot,
       width  = 45,
       height = 30,
       units  = "cm")




#### .... 변수갯수별 교차검증 결과 Summary ####
CrossValid_Summary_RFE_XGB <- Fit_RFE_XGB$CV_ResultSummary_Final %>% 
  .[, MODEL  := MODEL] %>% 
  .[, TARGET := TARGET_VAR] %>% 
  .[, METHOD := "XGB"] %>% 
  dplyr::select(MODEL, TARGET, METHOD, everything())

write.csv(CrossValid_Summary_RFE_XGB,
          file = sprintf("%s/%s_CrossValid_RFE_Summary_XGB_%s.csv", 
                         SAVE_FOLD, MODEL, gsub("-", "", Sys.Date())),
          row.names = FALSE,
          fileEncoding = "euc-kr")


#### .... 선택된 변수 ####
RFE_XGB_VAR_LS <- data.frame(MODEL    = MODEL,
                             TARGET   = TARGET_VAR,
                             METHOD   = "XGB",
                             VAR_NAME = VAR_SELECTED_LS)



#### .... RMSE 변화 그래프 ####
tmpDiffMinMax <- max(Fit_RFE_XGB$CV_ResultSummary_Final$TEST_RMSE_MEAN) - min(Fit_RFE_XGB$CV_ResultSummary_Final$TEST_RMSE_MEAN)

png(sprintf("%s/%s_Var_RFE_RMSE_PLOT_XGB_%s.png",
            SAVE_FOLD, MODEL, gsub("-", "", Sys.Date()))
)

RFE_XGB_CV_Result_Melt <- Fit_RFE_XGB$CV_ResultSummary_Final %>% 
  dplyr::select(SIZE, TRAIN_RMSE_MEAN, TEST_RMSE_MEAN) %>% 
  melt(id.vars = "SIZE") %>% 
  setnames(c("SIZE", "CATEGORY", "RMSE")) %>% 
  .[, CATEGORY := ifelse(grepl("TRAIN", CATEGORY), "TRAIN", "VALID")] %>% 
  .[, CATEGORY := base::factor(CATEGORY, levels = c("TRAIN", "VALID"))]

Variable_RMSE_Plot <- RFE_XGB_CV_Result_Melt %>% 
  ggplot() + 
  geom_line(aes(x    = SIZE,
                y    = RMSE,
                col  = CATEGORY)) +
  geom_text(aes(x     = SIZE,
                y     = RMSE + tmpDiffMinMax * 0.1,
                label = format(round(RMSE, 3)), digits = 3),
            data = RFE_XGB_CV_Result_Melt %>% 
              dplyr::filter(CATEGORY == "VALID") %>% 
              dplyr::filter(RMSE %in% c(max(RMSE), min(RMSE))),
            col = "red") +
  scale_color_manual(values = c("black", "red")) +
  labs(title = sprintf("%s's %s \n Vairable수에 따른 Validation RMSE 변화 in XGB", MODEL, TARGET_VAR), 30) +
  # ggtitle(label = sprintf("%s's %s : Vairable수에 따른 RMSE 변화 in XGB", MODEL, TARGET_VAR)) +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

print(Variable_RMSE_Plot)

dev.off()



#### .... MAE 변화 그래프 ####
tmpDiffMinMax <- max(Fit_RFE_XGB$CV_ResultSummary_Final$TEST_MAE_MEAN) - min(Fit_RFE_XGB$CV_ResultSummary_Final$TEST_MAE_MEAN)

png(sprintf("%s/%s_Var_RFE_MAE_PLOT_XGB_%s.png",
            SAVE_FOLD, MODEL, gsub("-", "", Sys.Date()))
)

RFE_XGB_MAE_CV_Result_Melt <- Fit_RFE_XGB$CV_ResultSummary_Final %>% 
  dplyr::select(SIZE, TRAIN_MAE_MEAN, TEST_MAE_MEAN) %>% 
  melt(id.vars = "SIZE") %>% 
  setnames(c("SIZE", "CATEGORY", "MAE")) %>% 
  .[, CATEGORY := ifelse(grepl("TRAIN", CATEGORY), "TRAIN", "VALID")] %>% 
  .[, CATEGORY := base::factor(CATEGORY, levels = c("TRAIN", "VALID"))]

Variable_MAE_Plot <- RFE_XGB_MAE_CV_Result_Melt %>% 
  ggplot() + 
  geom_line(aes(x    = SIZE,
                y    = MAE,
                col  = CATEGORY)) +
  geom_text(aes(x     = SIZE,
                y     = MAE + tmpDiffMinMax * 0.1,
                label = format(round(MAE, 3)), digits = 3),
            data = RFE_XGB_MAE_CV_Result_Melt %>% 
              dplyr::filter(CATEGORY == "VALID") %>% 
              dplyr::filter(MAE %in% c(max(MAE), min(MAE))),
            col = "red") +
  scale_color_manual(values = c("black", "red")) +
  labs(title = sprintf("%s's %s \n Vairable수에 따른 Validation MAE 변화 in XGB", MODEL, TARGET_VAR)) +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

print(Variable_MAE_Plot)

dev.off()



#### 7. Compare Valid's Real vs Pred ####
#### .. 7.2 XGB ####
RMSE_COMPARE_PLOT_RFE_XGB <- data.frame(
  READ = testData %>% select_(.dots = TARGET_VAR) %>% pull(),
  PRED = Pred_RFE_XGB %>% as.data.frame() %>% pull()
) %>% 
  ggplot(aes(x = READ, y = PRED)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  annotate("text",
           x = max(testData %>% select_(.dots = TARGET_VAR) %>% pull()) * 0.999,
           y = max(Pred_RFE_XGB) * 1.001,
           size = 3,
           label = sprintf("RMSE : %s", round(RMSE_RFE_XGB, 5)), 
           color = "red") +
  labs(x = "Real", y = "Pred", title = sprintf("%s's %s \n 실측값과 XGB of RFE 예측값 비교", 
                                               MODEL, TARGET_VAR)) +
  theme(plot.title = element_text(hjust = 0.5, size = 15)) +
  geom_smooth(method = "lm", se = TRUE)


#### 8. Compare Obs vs Pred with IDX ####
#### .. 8.2 XGB ####

## Train Set
png(sprintf("%s/%s_IDX_RFE_COMPARE_PLOT_XGB_%s.png",
            SAVE_FOLD, MODEL, gsub("-", "", Sys.Date()))
)

IDX_COMPARE_PLOT_RFE <- data.frame(
  IDX  = 1:NROW(trainData),
  REAL = trainData %>% select_(.dots = TARGET_VAR) %>% pull(),
  PRED =  predict(Fit_RFE_XGB$Final_Model, 
                  newdata = data.matrix(trainData %>% 
                                          dplyr::select_(.dots = VAR_SELECTED_LS))) %>% 
    as.data.frame() %>% pull()
) %>% 
  ggplot(aes(x = IDX, y = REAL)) +
  geom_point(col = "black", size = 1) +
  geom_point(aes(x = IDX, y = PRED), col = "blue", size = 1) +
  labs(x = "인덱스(순서)", y = "로딩량평균", 
       title = sprintf("%s's %s \n 실측값과 XGB of RFE 예측값 비교", 
                       MODEL, TARGET_VAR)) +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

print(IDX_COMPARE_PLOT_RFE)

dev.off()

## Valid Set
png(sprintf("%s/%s_IDX_RFE_VALID_COMPARE_PLOT_XGB_%s.png",
            SAVE_FOLD, MODEL, gsub("-", "", Sys.Date()))
)

IDX_COMPARE_VALID_PLOT_RFE <- data.frame(
  IDX  = 1:NROW(testData),
  REAL = testData %>% select_(.dots = TARGET_VAR) %>% pull(),
  PRED = Pred_RFE_XGB %>% as.data.frame() %>% pull()
) %>% 
  ggplot(aes(x = IDX, y = REAL)) +
  geom_point(col = "black", size = 1) +
  geom_point(aes(x = IDX, y = PRED), col = "blue", size = 1) +
  labs(x = "인덱스(순서)", y = "로딩량평균", 
       title = sprintf("%s's %s \n 실측값과 XGB of RFE 예측값 비교(Validation Set)", 
                       MODEL, TARGET_VAR)) +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

print(IDX_COMPARE_VALID_PLOT_RFE)

dev.off()

# RMSE(pred = Pred_RFE_XGB %>% as.data.frame() %>% pull(),
#      obs  = testData %>% select_(.dots = TARGET_VAR) %>% pull())

#### .... QQPLOT(Residual plot) ####
Test_Compare_DF_RAW <- data.table(
  TIME = testData$TIME,
  REAL = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull(),
  PRED_XGB = Pred_RFE_XGB
) 

write.csv(Test_Compare_DF_RAW,
          file = sprintf("%s/%s_Compare_Data_Raw_XGB_%s.csv",
                         SAVE_FOLD, MODEL,
                         gsub("-", "", TODAY)),
          row.names = FALSE,
          fileEncoding = "euc-kr")

Residual_Plot_RFE_XGB <- Test_Compare_DF_RAW %>% 
  .[, ERROR_XGB := REAL - PRED_XGB] %>% 
  melt(id.vars = "TIME") %>% 
  setnames(c("variable", "value"), c("CATEGORY", "VALUE")) %>% 
  .[grepl(pattern = "^ERROR", CATEGORY)] %>% 
  ggplot() +
  geom_point(aes(x = TIME, y = VALUE)) +
  geom_hline(yintercept = 0, col = "red") +
  scale_x_datetime(labels = date_format("%Y-%m-%d %H:%M:%S"), 
                   date_breaks = "6 hours", date_minor_breaks = "2 hours") +
  labs(title = "Residual Plot(IDX vs Error Term)",
       x = "IDX", y = "Error") +
  theme(plot.title = element_text(hjust = 0.5, size = 15),
        axis.text.x = element_blank(),
        legend.position = "bottom") +
  facet_wrap(~CATEGORY,
             nrow = 2
             # , scale = "free"
  )
print(Residual_Plot_RFE_XGB)  

## Save
ggsave(file   = sprintf("%s/%s_Residual_Plot_RFE_XGB__%s.png", 
                        SAVE_FOLD, MODEL, gsub("-", "", TODAY)),
       plot   = Residual_Plot_RFE_XGB,
       width  = 45,
       height = 30,
       units  = "cm")




END_TIME <- Sys.time()

cat("\n",
    sprintf("===== Total Process is END! ====="),
    '\n Start : ', paste0(ST_TIME),
    '\n End   : ', paste0(END_TIME),
    '\n',
    capture.output(difftime(END_TIME, ST_TIME, units = "mins")),
    "\n",
    "\n")

# save.image("~/Loading2/03.result/RFE/20200214/SAVE_RData_20200215.RData")