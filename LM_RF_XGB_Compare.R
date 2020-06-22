
####################################################################################
## GLM vs RF vs XGB 결과 비교
## Test set 이동하면서 결과 비교
## 변수 조합 4가지
####################################################################################

#### 0. Set Options ####
rm(list = ls())
gc()

options(scipen = 100, max.print = 5000, stringsAsFactors = FALSE,
        repos = c(CRAN = "http://cran.rstudio.com"))

DEFALUT_FOLD <- ".."
setwd(DEFALUT_FOLD)

TODAY         <- Sys.Date()
TOTAL_ST_TIME <- Sys.time()

# #### .. 저장경로 ####
SAVE_FOLD <- sprintf("03.result/COMPARE_GLM_RF_XGB/%s", gsub("-", "", TODAY))
if( !dir.exists(SAVE_FOLD) ) { dir.create(SAVE_FOLD, recursive = TRUE) }

#### 1. library load ####
package <- c("data.table", "tidyverse", "DescTools", "lubridate", "scales", "ggrepel", "gridExtra", "RJDBC",
             "caret", "h2o", "glmnet", "elasticnet", "lm.beta", 
             "randomForest", "xgboost", "SHAPforxgboost", "progress",
             "foreach", "parallel", "doParallel", "doSNOW")
sapply(package, require, character.only = TRUE)

filter     <- dplyr::filter
lag        <- dplyr::lag
wday       <- lubridate::wday
month      <- lubridate::month
week       <- data.table::week
between    <- dplyr::between
row_number <- dplyr::row_number

MAE        <- caret::MAE
RMSE       <- caret::RMSE


## Set time zone
Sys.setenv(TZ = "Asia/Seoul")
Sys.timezone(location = TRUE)

# ## 시스템 사양
# NCmisc::top(CPU = FALSE, RAM = TRUE)
# NCmisc::memory.summary(unit = "mb")
# parallel::detectCores()


#####################
#### Start ROOF! ####
##################### 

#### 2. Data Load ####
analData    <- fread("01.data/dataset.csv")

VAR_LS           <- colnames(analData)[5:NCOL(analData)]
VAR_LS           <- VAR_LS[!grepl(pattern = "^LOAD_TOP|WEB_TARGET_TOP", VAR_LS)]
TARGET_VAR       <- "LOAD_MEAN"

#### .. Filter Needed columns ####
analData <- analData %>% 
  .[, TIME := lubridate::ymd_hms(TIME, tz = "Asia/Seoul")] %>% 
  dplyr::select_(.dots = c("LOT_ID", "DATE", "TIME", VAR_LS, TARGET_VAR)) %>% 
  .[order(TIME)]

analData[, GAP_PLUS := GAP_L + GAP_R]
analData$GAP_L <- NULL
analData$GAP_R <- NULL
VAR_LS <- setdiff(VAR_LS, c("GAP_L", "GAP_R"))
VAR_LS <- c(VAR_LS, "GAP_PLUS")


#### 3. Define User's Function & Variable ####
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
#####################################################################################


#### * Set Variable ####

CORE_N            <- 12
MODEL             <- "M48F"
P_VALUE_CUTOFF    <- 0.05
CORR_CUTOFF       <- 0.9
EARLY_STOP_NROUND <- 15
analData_RAW      <- analData

PRED_MIN <- 0
PRED_MAX <- (analData_RAW %>% dplyr::select_(.dots = TARGET_VAR) %>% pull() %>% max()) * 1.2

VISCOSITY_LS <- paste0("VISCOSITY_", 1:4)
TEST_SET_LS  <- c(unique(analData_RAW$DATE), "Random")
VAR_COMB_LS  <- c("A", "B", "C", "ALL")

VAR_LS_RAW   <- setdiff(VAR_LS, c(VISCOSITY_LS, "ERR_TYPE", TARGET_VAR))

h2o.init(nthreads = CORE_N)
h2o.no_progress()

#### * ROOF Start! ####

for( vs0 in 1:NROW(VISCOSITY_LS) ){
  
  # vs0 <- 1
  
  if( vs0 == 1 ){ 
    ERROR_Result      <- NULL
    VAR_Import_Result <- NULL
    GLM_Use_Var       <- NULL
    baseLine_ERROR    <- NULL
  }
  
  VISCOSITY_TARGET <- VISCOSITY_LS[vs0]
  VAR_LS           <- c(VAR_LS_RAW, VISCOSITY_TARGET)
  
  analData <- analData_RAW %>% 
    dplyr::filter(!is.na(!!sym(VISCOSITY_TARGET))) %>% 
    setDT()
  
  for( t00 in 1:NROW(TEST_SET_LS) ){
    
    # t00 <- 1
    
    TEST_SET <- TEST_SET_LS[t00]
    
    #### 4. Set Train / Valid  ####
    if ( TEST_SET != "Random" ){
      trainData <- analData[DATE != TEST_SET]
      testData  <- analData[DATE == TEST_SET]
    } else {
      set.seed(2020)
      TRAIN_IDX <- createDataPartition(analData$DATE, p = 0.8)$Resample1
      trainData <- analData[TRAIN_IDX]
      testData  <- analData[-TRAIN_IDX]
    }
    
    baseLine_ERROR <- rbind(
      baseLine_ERROR,
      data.table(VISCOSITY  = VISCOSITY_TARGET,
                 TEST_SET   = TEST_SET,
                 MAE        = MAE(obs = testData %>% select_(.dots = TARGET_VAR) %>% pull(),
                                  pred = mean(testData %>% select_(.dots = TARGET_VAR) %>% pull())),
                 RMSE       = RMSE(obs = testData %>% select_(.dots = TARGET_VAR) %>% pull(),
                                   pred = mean(testData %>% select_(.dots = TARGET_VAR) %>% pull())),
                 TRAIN_NROW = NROW(trainData),
                 TEST_NROW  = NROW(testData)
      )
    )
    
    
    for( v00 in 1:NROW(VAR_COMB_LS) ){
      
      # v00 <- 4
      
      VAR_COMB <- VAR_COMB_LS[v00]
      
      cat(sprintf("\n==== %s ====\n==== (%s/%s) Test_Set:%s / VAR_COMB:%s ==== \n==== %s ====",
                  VISCOSITY_TARGET,
                  v00 + (t00 - 1) * NROW(VAR_COMB_LS), 
                  NROW(TEST_SET_LS) * NROW(VAR_COMB_LS), 
                  TEST_SET, VAR_COMB, Sys.time()),
          "\n",
          "\n")
      
      #### 5. 변수 조합 ####
      if ( VAR_COMB == "A" ){
        VAR_LS <- c("GAP_PLUS", "SLURRY_PUMP_SP")
      } else if ( VAR_COMB == "B" ){
        VAR_LS <- c("GAP_PLUS", "SLURRY_PUMP_SP", VISCOSITY_TARGET)
      } else if ( VAR_COMB == "C" ){
        VAR_LS <- c("GAP_PLUS", "SLURRY_PUMP_SP", VISCOSITY_TARGET, "VISCOSITY_TEMP")
      } else if ( VAR_COMB == "ALL" ){
        set.seed(2020)
        
        NEEDED_VAR <- c("GAP_PLUS", "SLURRY_PUMP_SP", VISCOSITY_TARGET, "VISCOSITY_TEMP")
        
        preProc <- preProcess(trainData %>% 
                                dplyr::select_(.dots = VAR_LS_RAW[!grepl(paste(NEEDED_VAR, collapse = "|"),
                                                                         VAR_LS_RAW)]),
                              method      = c("zv", "corr"),
                              cutoff      = CORR_CUTOFF,
                              verbose     = FALSE)
        
        VAR_LS <- VAR_LS_RAW[!grepl(pattern = paste(preProc$method$remove, collapse = "|"),
                                    VAR_LS_RAW)]
        
        VAR_LS <- c(VAR_LS, NEEDED_VAR) %>% unique
      }
      
      
      #### 6. Fit Modeling ####
      
      #### ..6.1. GLM | LASSO ####
      
      if ( VAR_COMB != "ALL" ){
        
        cat("==== Fit GLM Start! ====", "\n")
        if( any(grepl(VISCOSITY_TARGET, VAR_LS)) ){
          
          ## 상호작용을 위한 Formula
          tmpVAR_LS <- VAR_LS[!grepl(VISCOSITY_TARGET, VAR_LS)]
          LM_FORMULA_FIRST <- as.formula(sprintf("%s ~ %s + %s", 
                                                 TARGET_VAR,
                                                 paste(tmpVAR_LS, collapse = " + "),
                                                 paste(paste(tmpVAR_LS, ":", VISCOSITY_TARGET), collapse = " + ")
          ))
          
          Fit_LM_First <- train(LM_FORMULA_FIRST,
                                data   = trainData,
                                method = "lm",
                                metric = "RMSE",
                                trControl = trainControl(method  = "repeatedcv",
                                                         number  = 5,
                                                         repeats = 3,
                                                         allowParallel = FALSE),
                                tuneGrid = expand.grid(intercept = FALSE))
          
          P_Value <- summary(Fit_LM_First)$coefficients %>% 
            as.data.frame() %>% 
            rownames_to_column() %>% 
            setnames(c("VAR_NAME", "ESTIMATE", "SD", "T_VALUE", "P_VALUE")) %>% 
            dplyr::mutate(VAR_NAME = gsub("`", "", VAR_NAME))
          
          SELECTED_VAR <- P_Value %>% 
            dplyr::filter(P_VALUE < P_VALUE_CUTOFF) %>% 
            dplyr::filter(!grepl("Intercept", VAR_NAME)) %>% 
            .$VAR_NAME
          
          LM_FORMULA <- as.formula(sprintf("%s ~ %s", 
                                           TARGET_VAR,
                                           paste(SELECTED_VAR, collapse = " + ")
          ))
          
          Fit_LM <- train(LM_FORMULA,
                          data   = trainData,
                          method = "lm",
                          metric = "RMSE",
                          trControl = trainControl(method  = "repeatedcv",
                                                   number  = 5,
                                                   repeats = 3,
                                                   allowParallel = FALSE),
                          tuneGrid = expand.grid(intercept = FALSE))
          
          P_Value2 <- summary(Fit_LM)$coefficients %>% 
            as.data.frame() %>% 
            rownames_to_column() %>% 
            setnames(c("VAR_NAME", "ESTIMATE", "SD", "T_VALUE", "P_VALUE")) %>% 
            dplyr::mutate(VAR_NAME = gsub("`", "", VAR_NAME))
          
          SD_ESTIMTAE <- lm.beta(Fit_LM$finalModel)$standardized.coefficients %>% 
            as.data.frame() %>% 
            rownames_to_column() %>% 
            setnames(c("VAR_NAME", "STANDARD_ESTIMATE")) %>%
            dplyr::mutate(VAR_NAME = gsub("`", "", VAR_NAME))
          
          
          # GLM_Use_Var <- NULL
          GLM_Use_Var <- rbindlist(list(GLM_Use_Var,
                                        data.table(VISCOSITY = VISCOSITY_TARGET,
                                                   TEST_SET  = TEST_SET,
                                                   VAR_COMB  = VAR_COMB,
                                                   P_Value2 %>% dplyr::select(VAR_NAME, ESTIMATE, P_VALUE)) %>% 
                                          merge(SD_ESTIMTAE,
                                                by = "VAR_NAME") %>% 
                                          dplyr::select(VISCOSITY, TEST_SET, VAR_COMB, everything())),
                                   fill = TRUE)
          
        } else {
          
          LM_FORMULA <- as.formula(sprintf("%s ~ %s", 
                                           TARGET_VAR,
                                           paste(VAR_LS, collapse = " + ")
          ))
          
          Fit_LM <- train(LM_FORMULA,
                          data   = trainData %>% dplyr::select_(.dots = c(TARGET_VAR, VAR_LS)),
                          method = "lm",
                          metric = "RMSE",
                          trControl = trainControl(method  = "repeatedcv",
                                                   number  = 5,
                                                   repeats = 3,
                                                   allowParallel = FALSE),
                          tuneGrid = expand.grid(intercept = FALSE))
          
          P_Value <- summary(Fit_LM)$coefficients %>% 
            as.data.frame() %>% 
            rownames_to_column() %>% 
            setnames(c("VAR_NAME", "ESTIMATE", "SD", "T_VALUE", "P_VALUE")) %>% 
            dplyr::mutate(VAR_NAME = gsub("`", "", VAR_NAME))
          
          
          SD_ESTIMTAE <- lm.beta(Fit_LM$finalModel)$standardized.coefficients %>% 
            as.data.frame() %>% 
            rownames_to_column() %>% 
            setnames(c("VAR_NAME", "STANDARD_ESTIMATE")) %>%
            dplyr::mutate(VAR_NAME = gsub("`", "", VAR_NAME))
          
          GLM_Use_Var <- rbindlist(list(GLM_Use_Var,
                                        data.table(VISCOSITY = VISCOSITY_TARGET,
                                                   TEST_SET  = TEST_SET,
                                                   VAR_COMB  = VAR_COMB,
                                                   P_Value %>% dplyr::select(VAR_NAME, ESTIMATE, P_VALUE)) %>% 
                                          merge(SD_ESTIMTAE,
                                                by = "VAR_NAME") %>% 
                                          dplyr::select(VISCOSITY, TEST_SET, VAR_COMB, everything())),
                                   fill = TRUE)
          
        } 
      } else {
        
        cat("==== Fit LASSO Regression Start! ====", "\n")
        
        NEEDED_VAR <- c("GAP_PLUS", "SLURRY_PUMP_SP", VISCOSITY_TARGET, "VISCOSITY_TEMP")
        tmpVAR_LS  <- NEEDED_VAR[!grepl(VISCOSITY_TARGET, NEEDED_VAR)]
        LM_FORMULA <- as.formula(sprintf("%s ~ %s + %s + %s", 
                                         TARGET_VAR,
                                         paste(tmpVAR_LS, collapse = " + "),
                                         paste(paste(tmpVAR_LS, ":", VISCOSITY_TARGET), collapse = " + "),
                                         paste(setdiff(VAR_LS, NEEDED_VAR), collapse = " + ")
        ))
        
        
        if( CORE_N > 1 ){
          CORE_N      <- min(CORE_N, parallel::detectCores() - 1)
          MainCluster <- makeCluster(CORE_N)
          registerDoSNOW(MainCluster)
          
          PARALLEL_TF <- TRUE
        } else {
          PARALLEL_TF <- FALSE
        }
        
        Fit_LM <- train(LM_FORMULA,
                        data   = trainData %>% dplyr::select_(.dots = c(TARGET_VAR, VAR_LS)),
                        method = "lasso",
                        metric = "RMSE",
                        trControl = trainControl(method  = "repeatedcv",
                                                 number  = 5,
                                                 repeats = 5,
                                                 allowParallel = PARALLEL_TF
                        ),
                        tuneGrid = expand.grid(fraction = seq(from = 0.1, to = 0.9, by = 0.02)),
                        # tuneLength = 10,
        )
        
        if( CORE_N > 1 ){
          stopCluster(MainCluster)
        }
        
        LASSO_Coef <- predict(Fit_LM$finalModel,
                              s    = Fit_LM$bestTune[1, "fraction"],
                              type = "coefficients",
                              mode = "fraction")$coefficients %>% 
          as.data.frame() %>% 
          rownames_to_column() %>% 
          setDT() %>% 
          .[, P_VALUE := Fit_LM$bestTune[1, "fraction"]] %>% 
          setnames(c("VAR_NAME", "ESTIMATE", "P_VALUE"))
        
        GLM_Use_Var <- rbindlist(list(GLM_Use_Var,
                                      data.table(VISCOSITY = VISCOSITY_TARGET,
                                                 TEST_SET  = TEST_SET,
                                                 VAR_COMB  = VAR_COMB,
                                                 LASSO_Coef)),
                                 fill = TRUE)
        
        #### .... Fraction Plot ####
        
        tmpDiffMinMax      <- max(Fit_LM$results$RMSE) - min(Fit_LM$results$RMSE)
        LASSO_FractionPlot <- Fit_LM$results %>% 
          ggplot() +
          geom_point(aes(x = fraction, y = RMSE), shape = 1, col = "blue") +
          labs(title = sprintf("LASSO Regression's Fraction Plot \n %s \n Test:%s / Variable Comb:%s", 
                               VISCOSITY_TARGET,
                               TEST_SET, VAR_COMB),
               x = "Fraction of Full Solution",
               y = "RMSE(Repeated Cross-Validation)") +
          geom_text(aes(x = fraction, 
                        y = RMSE + tmpDiffMinMax * 0.1,
                        label = format(round(RMSE, 4)), digits = 4),
                    data = Fit_LM$results %>% dplyr::filter(RMSE %in% c(min(RMSE), max(RMSE))),
                    col = "red") +
          theme(plot.title = element_text(hjust = 0.5, size = 15))
        
        ggsave(file   = sprintf("%s/%s_LASSO_Fraction_Plot_Test_%s_VarComb_%s__%s.png",
                                SAVE_FOLD, VISCOSITY_TARGET, TEST_SET, VAR_COMB, gsub("-", "", TODAY)),
               plot   = LASSO_FractionPlot,
               width  = 45,
               height = 30,
               units  = "cm")
        
        
        #### .... Coefficient Plot ####
        png(filename = sprintf("%s/%s_LASSO_Coefficients_Plot_Test_%s_VarComb_%s__%s.png",
                               SAVE_FOLD, VISCOSITY_TARGET, TEST_SET, VAR_COMB, gsub("-", "", TODAY)))
        
        plot(Fit_LM$finalModel,
             main = sprintf("LASSO Regression's Coefficients Plot \n %s \n Test:%s / Variable Comb:%s", 
                            TEST_SET, VISCOSITY_TARGET, VAR_COMB)
        )
        
        dev.off()
        
      }
      
      
      #### .... Predict ####
      Pred_LM <- predict(Fit_LM, newdata = testData)
      Pred_LM <- sapply(Pred_LM, function(x) { max(min(x, PRED_MAX), PRED_MIN) })
      
      RMSE_LM <- RMSE(pred = Pred_LM, 
                      obs  = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
      MAE_LM  <- MAE(pred = Pred_LM, 
                     obs  = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
      
      
      #### ..6.2 RF ####
      cat("==== Fit RF Start! ====", "\n")
      
      NTREE <- 300
      
      Fit_RF <- h2o.randomForest(
        x                                 = VAR_LS,
        y                                 = TARGET_VAR,
        training_frame                    = as.h2o(trainData %>% dplyr::select_(.dots = c(VAR_LS, TARGET_VAR))),
        validation_frame                  = as.h2o(testData  %>% dplyr::select_(.dots = c(VAR_LS, TARGET_VAR))), 
        seed                              = 2020,
        stopping_metric                   = "RMSE",
        stopping_rounds                   = EARLY_STOP_NROUND,
        stopping_tolerance                = 1e-2 * 1,
        score_tree_interval               = 30,
        ntree                             = NTREE,
        verbose                           = FALSE
      )
      
      #### .... Predict ####
      Pred_RF <- predict(Fit_RF, newdata = as.h2o(testData %>% dplyr::select_(.dots = c(VAR_LS, TARGET_VAR))))
      Pred_RF <- Pred_RF %>% as.data.frame() %>% pull() %>% sapply(function(x) { max(min(x, PRED_MAX), PRED_MIN) })
      
      RMSE_RF <- RMSE(pred = Pred_RF,
                      obs  = testData %>% select_(.dots = TARGET_VAR) %>% pull())
      MAE_RF  <- MAE(pred = Pred_RF,
                     obs  = testData %>% select_(.dots = TARGET_VAR) %>% pull())
      
      #### .... Variable Importance ####
      VariableImportance_RF <- h2o.varimp(Fit_RF) %>%
        as.data.frame() %>%
        select(variable, scaled_importance) %>%
        rename(VAR_NAME   = variable,
               VAR_IMPORT = scaled_importance) %>%
        dplyr::mutate(VISCOSITY = VISCOSITY_TARGET,
                      TEST_SET  = TEST_SET,
                      VAR_COMB  = VAR_COMB,
                      METHOD    = "RF") %>%
        dplyr::select(VISCOSITY, TEST_SET, VAR_COMB, METHOD, everything()) %>%
        dplyr::mutate(VAR_IMPORT = VAR_IMPORT / max(VAR_IMPORT)) %>%
        dplyr::arrange(VISCOSITY, TEST_SET, VAR_COMB, desc(VAR_IMPORT), VAR_NAME)
      
      # write.csv(VariableImportance_RF,
      #           file = sprintf("%s/Var_Import_RF_Test_%s_VarComb_%s_%s.csv",
      #                          SAVE_FOLD, gsub("-", "", TEST_SET), VAR_COMB, gsub("-", "", Sys.Date())),
      #           row.names = FALSE,
      #           fileEncoding = "euc-kr")
      
      
      # Fit_RF@model$scoring_history %>%
      #   select(number_of_trees, training_rmse, validation_rmse) %>%
      #   melt(id = c("number_of_trees")) %>%
      #   rename(RMSE = value, NUM_TREE = number_of_trees, CATEGORY = variable) %>%
      #   dplyr::mutate(CATEGORY = case_when(grepl("train", CATEGORY) ~ "Train",
      #                                      TRUE ~ "Valid")) %>%
      #   ggplot() +
      #   geom_line(aes(x = NUM_TREE, y = RMSE, col = CATEGORY, fill = CATEGORY)) +
      #   labs(title = "RF's RMSE of Train vs Valid ") +
      #   theme(legend.position = "bottom",
      #         plot.title = element_text(hjust = 0.5, size = 15))
      
      #### ..6.3 XGB ####
      #### .... Grid Seach for Parameter ####
      
      cat("==== Fit XGB Start! ====", "\n")
      NTREE <- 1000
      
      PARAGRID_LS <- expand.grid(subsample        = c(0.7), 
                                 colsample_bytree = c(0.5, 0.7),
                                 max_depth        = c(4, 6, 8, 10),
                                 min_child        = 1, 
                                 eta              = c(0.1, 0.01)
      )
      
      #### .... Execute Grid Search ####
      if( NROW(PARAGRID_LS) > 1 ){
        
        if( CORE_N > 1 ){
          CORE_CNT     <- min(3, parallel::detectCores() - 1)
          MAIN_CLUSTER <- makeCluster(CORE_CNT)
          registerDoSNOW(MAIN_CLUSTER)
          
          CORE_SUB_CNT <- floor(CORE_N / CORE_CNT)
        } else {
          CORE_SUB_CNT <- 1
        }
        
        # PB       <- txtProgressBar(max = NROW(FILE_LS), style = 3)
        # PROGRESS <- function(n) { setTxtProgressBar(PB, n) }
        
        IterName       <- apply(PARAGRID_LS, 1, function(x) 
          sprintf("Subsample:%s / Colsample:%s / MaxDepth:%s / eta:%s", x[1], x[2], x[3], x[5])
        )
        
        RemainIterName <- IterName
        EndIterCnt     <- 0
        TotalIterCNT   <- NROW(IterName)
        ST_TIME        <- Sys.time()
        
        PROGRESS       <- function(n){
          
          EndIterCnt     <<- EndIterCnt + 1
          RemainIterName <<- setdiff(RemainIterName, IterName[n])
          
          cat(sprintf("==== %s / %s(%s) is End with %s Mins! ====", 
                      EndIterCnt, TotalIterCNT, IterName[n], 
                      round(as.numeric(difftime(Sys.time(), ST_TIME, units = "mins")), 1)), 
              "\n")
          
          # if ( (TotalIterCNT - EndIterCnt) <= 5 & (TotalIterCNT - EndIterCnt) >= 1 ){
          #   cat(sprintf("==== Remain Process : %s ====", paste(RemainIterName, collapse = ", ")), "\n", "\n")
          # }
          
          if ( TotalIterCNT == EndIterCnt ){
            cat(sprintf("==== CV Process is End! %s ====", Sys.time()), "\n", "\n")
          }
        }
        ######################################################################
        
        OPTS     <- list(progress = PROGRESS)
        
        paraResult <- foreach(i0 = 1:NROW(PARAGRID_LS)
                              , .combine = function(x, y) { rbindlist(list(x, y), fill = TRUE) }
                              , .errorhandling = "stop"
                              # , .options.snow  = OPTS
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
          
          tmpXgboost <- xgb.cv(
            data                  = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                                label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
            seed                  = 2020,
            nround                = NTREE,
            subsample             = tmpSubsample,
            colsample_bytree      = tmpColsample,
            max_depth             = tmpMaxdepth,
            min_child_weight      = tmpMinchild,
            eta                   = tmpEta,
            nthread               = CORE_SUB_CNT,
            early_stopping_rounds = EARLY_STOP_NROUND,
            nfold                 = 5,
            verbose               = 0,
            metrics               = "rmse"
          )
          
          # ## Train vs Valid's RMSE Plot
          # tmpXboost_RMSE_Log <- tmpXgboost$evaluation_log %>%
          #   dplyr::select(iter, train_rmse_mean, test_rmse_mean) %>%
          #   reshape2::melt(id.vars = "iter") %>%
          #   rename(Iter = iter, Category = variable, RMSE = value)
          # 
          # tmpXboost_RMSE_Log %>%
          #   # dplyr::filter(RMSE <= 1) %>%
          #   ggplot() +
          #   geom_point(aes(x = Iter, y = RMSE, group = Category, col = Category),
          #              alpha = 0.5, size = 0.1)
          
          BEST_ITER       <- tmpXgboost$best_iteration
          ## Train
          TRAIN_RMSE_MEAN <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$train_rmse_mean
          TRAIN_RMSE_SD   <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$train_rmse_std
          ## Valid
          VALID_RMSE_MEAN <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$test_rmse_mean
          VALID_RMSE_SD   <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$test_rmse_std
          
          ## Result
          tmpRMSE_Result <- data.table(
            SUBSAMPLE             = tmpSubsample,
            COLSAMPLE_BYTREE      = tmpColsample,
            MAX_DEPTH             = tmpMaxdepth,
            MIN_CHILD_WEIGHT      = tmpMinchild,
            ETA                   = tmpEta,
            BEST_ITER             = BEST_ITER,
            TRAIN_RMSE_MEAN       = TRAIN_RMSE_MEAN,
            TRAIN_RMSE_SD         = TRAIN_RMSE_SD,
            VALID_RMSE_MEAN       = VALID_RMSE_MEAN,
            VALID_RMSE_SD         = VALID_RMSE_SD
          )
          
          return(tmpRMSE_Result)
        }
        
        if( CORE_N > 1){
          stopCluster(MAIN_CLUSTER)
        }
        
        paraResult <- paraResult %>% dplyr::arrange(VALID_RMSE_MEAN, VALID_RMSE_SD)
        
        bestTunePara <- paraResult %>% 
          dplyr::filter(VALID_RMSE_MEAN == min(VALID_RMSE_MEAN)) %>% 
          dplyr::arrange(VALID_RMSE_MEAN, VALID_RMSE_SD) %>% .[1, ]
        
        SUBSAMPLE         <- bestTunePara$SUBSAMPLE
        COLSAMPLE_BYTREE  <- bestTunePara$COLSAMPLE_BYTREE
        MAX_DEPTH         <- bestTunePara$MAX_DEPTH
        MIN_CHILD_WEIGHT  <- bestTunePara$MIN_CHILD_WEIGHT
        ETA               <- bestTunePara$ETA
        BEST_ITER         <- round(bestTunePara$BEST_ITER, -2)
        
        Fit_XGB <- xgb.train(
          data                  = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                              label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
          seed                  = 2020,
          nround                = BEST_ITER,
          subsample             = SUBSAMPLE,
          colsample_bytree      = COLSAMPLE_BYTREE,
          max_depth             = MAX_DEPTH,
          min_child_weight      = MIN_CHILD_WEIGHT,
          eta                   = ETA,
          nthread               = CORE_N,
          early_stopping_rounds = EARLY_STOP_NROUND,
          verbose               = 0, 
          watchlist             = list(
            train = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
            test = xgb.DMatrix(data  = data.matrix(testData %>% dplyr::select_(.dots = VAR_LS)),
                               label = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
          )
        )
        
      } else {
        
        ## Do not Grid Search 
        Fit_XGB <- xgb.train(
          data                  = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                              label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
          seed                  = 2020,
          nround                = NTREE,
          subsample             = PARAGRID_LS$subsample,
          colsample_bytree      = PARAGRID_LS$colsample_bytree,
          max_depth             = PARAGRID_LS$max_depth,
          min_child_weight      = PARAGRID_LS$min_child,
          eta                   = PARAGRID_LS$eta,
          nthread               = CORE_N,
          early_stopping_rounds = EARLY_STOP_NROUND,
          verbose               = 0, 
          watchlist             = list(
            train = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
            test = xgb.DMatrix(data  = data.matrix(testData %>% dplyr::select_(.dots = VAR_LS)),
                               label = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
          )
          
        )
        
      } ## End of XGB 
      ########################################################
      
      # Fit_XGB$evaluation_log %>% 
      #   melt(id.vars = "iter") %>% 
      #   setnames(c("ITER", "CATEGORY", "VALUE")) %>% 
      #   .[VALUE < 10] %>% 
      #   ggplot() +
      #   geom_point(aes(x = ITER, y = VALUE, col = CATEGORY, fill = CATEGORY))
      
      
      #### .... Predict ####
      Pred_XGB <- predict(Fit_XGB,
                          newdata = data.matrix(testData %>% dplyr::select_(.dots = VAR_LS)))
      Pred_XGB <- Pred_XGB %>% sapply(function(x) { max(min(x, PRED_MAX), PRED_MIN) })
      
      RMSE_XGB <- RMSE(pred = Pred_XGB,
                       obs  = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull())
      MAE_XGB  <- MAE(pred = Pred_XGB,
                      obs  = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull())
      
      #### .... Variable Importance ####
      VariableImportance_XGB <- xgb.importance(model = Fit_XGB) %>% 
        as.data.frame() %>% 
        select(Feature, Gain) %>% 
        setnames(c("VAR_NAME", "VAR_IMPORT")) %>% 
        dplyr::mutate(VAR_IMPORT = VAR_IMPORT / max(VAR_IMPORT)) %>% 
        dplyr::mutate(VISCOSITY = VISCOSITY_TARGET,
                      TEST_SET  = TEST_SET,
                      VAR_COMB  = VAR_COMB,
                      METHOD    = "XGB") %>% 
        dplyr::select(VISCOSITY, TEST_SET, VAR_COMB, METHOD, VAR_NAME, VAR_IMPORT, everything())
      
      VariableImportance_XGB <- data.table(VAR_NAME = VAR_LS) %>% 
        merge(VariableImportance_XGB,
              by = "VAR_NAME",
              all.x = TRUE) %>% 
        replace_na(list(VISCOSITY  = VISCOSITY_TARGET,
                        TEST_SET   = TEST_SET,
                        VAR_COMB   = VAR_COMB,
                        METHOD     = "XGB",
                        VAR_IMPORT = 0)) %>% 
        .[order(desc(VAR_IMPORT))] %>% 
        dplyr::select(VISCOSITY, TEST_SET, VAR_COMB, METHOD, VAR_NAME, VAR_IMPORT, everything())
      
      # write.csv(VariableImportance_XGB,
      #           file = sprintf("%s/Var_Import_XGB_Test_%s_VarComb_%s_%s.csv",
      #                          SAVE_FOLD, gsub("-", "", TEST_SET), VAR_COMB, gsub("-", "", Sys.Date())),
      #           row.names = FALSE,
      #           fileEncoding = "euc-kr")
      
      
      #### .... SHAP Value ####
      cat("==== Fit XGB's SHAP Value Start! ====", "\n")
      # ## To return the SHAP values and ranked features by mean|SHAP|
      # shap_Values <- shap.values(xgb_model = FitXgboost,
      #                            X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))
      # ## The ranked features by mean |SHAP|
      # shap_Values$mean_shap_score
      
      
      ## To prepare the long-format data:
      shap_Long <- shap.prep(xgb_model = Fit_XGB,
                             X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))
      
      # #### .. SHAP summary plot ####
      # shap.plot.summary(shap_Long)
      # 
      # 
      # #### .. diluted points ####
      # shap.plot.summary(shap_Long, x_bound  = 1.2, dilute = 10)
      # 
      # 
      # #### .. Alternatives ways to make the same plot ####
      # #### .... Option 1: from the xgboost model ####
      # shap.plot.summary.wrap1(xgb_model = FitXgboost,
      #                         X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))
      # 
      # 
      # #### .... Option 2: supply a self-made SHAP values dataset  ####
      # ## (e.g. sometimes as output from cross-validation)
      # shap.plot.summary.wrap2(shap_Values$shap_score,
      #                         as.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)))
      
      
      
      #### .. Dependency Plot ####
      # ## Prepare the data
      # shap_Int <- shap.prep.interaction(xgb_model = FitXgboost,
      #                                   X_train   = trainData %>% dplyr::select_(.dots = VAR_LS))
      
      
      ## SHAP Interaction effect plot
      
      for( v0 in 1:NROW(VAR_LS) ){
        
        # v0 <- 1
        
        if( v0 == 1 ){
          
          SAVE_FOLD_SHAP <- sprintf("%s/SHAP/Dependency_Plot/%s/TESTSET_%s/VAR_COMB_%s", 
                                    SAVE_FOLD, VISCOSITY_TARGET, gsub("-", "", TEST_SET), VAR_COMB)
          if( !dir.exists(SAVE_FOLD_SHAP) ) { dir.create(SAVE_FOLD_SHAP, recursive = TRUE) }
          
          # ## Set Progress bar
          # varPB <- progress_bar$new(total = NROW(VAR_LS))
        }
        
        iterVar <- VAR_LS[v0]
        
        tmpDepend_Plot <- shap.plot.dependence(data_long = shap_Long,
                                               x = iterVar,
                                               y = iterVar,
                                               color_feature = VISCOSITY_TARGET,
                                               add_hist      = TRUE,
                                               smooth        = FALSE
        )
        
        ## Save
        ggsave(file   = sprintf("%s/%s__colorFeatur_%s__TestSet_%s__VarComb_%s__%s.png",
                                SAVE_FOLD_SHAP,
                                iterVar, VISCOSITY_TARGET, gsub("-", "", TEST_SET), VAR_COMB,
                                gsub("-", "", TODAY)),
               plot   = grid.arrange(tmpDepend_Plot),
               width  = 45,
               height = 30,
               units  = "cm")
        
        # ## Show Progress bar
        # varPB$tick()
        
      }
      
      
      #### .. 6.4. XGB_Linear ####
      cat("==== Fit XGB Linear Start! ====", "\n")
      PARAGRID_LINEARLS <- expand.grid(lambda = seq(from = 0, to = 0.9, by = 0.1),
                                       alpha  = seq(from = 0, to = 0.9, by = 0.1))
      
      PARAGRID_LINEARLS <- PARAGRID_LINEARLS %>% 
        sample_n(floor(NROW(PARAGRID_LINEARLS) / 10)) %>% 
        dplyr::arrange(lambda, alpha)
      
      NTREE <- 500
      
      #### .... Execute Grid Search ####
      if( NROW(PARAGRID_LINEARLS) > 1 ){
        
        if( CORE_N > 1 ){
          CORE_CNT     <- min(3, parallel::detectCores() - 1)
          MAIN_CLUSTER <- makeCluster(CORE_CNT)
          registerDoSNOW(MAIN_CLUSTER)
          
          CORE_SUB_CNT <- floor(CORE_N / CORE_CNT)
        } else {
          CORE_SUB_CNT <- 1
        }
        
        # PB       <- txtProgressBar(max = NROW(FILE_LS), style = 3)
        # PROGRESS <- function(n) { setTxtProgressBar(PB, n) }
        # OPTS     <- list(progress = PROGRESS)
        
        paraResult_Linear <- foreach(i0 = 1:NROW(PARAGRID_LINEARLS)
                                     , .combine = function(x, y) { rbindlist(list(x, y), fill = TRUE) }
                                     , .errorhandling = "stop"
                                     # , .options.snow  = OPTS
                                     , .packages      = c("data.table", "tidyverse", "lubridate", "scales", 
                                                          "caret", "xgboost", "SHAPforxgboost",
                                                          "foreach", "parallel", "doParallel", "doSNOW")
        ) %dopar% {
          
          # i0 <- 1
          
          tmpLambda <- PARAGRID_LINEARLS[i0, ]$lambda
          tmpAlpha  <- PARAGRID_LINEARLS[i0, ]$alpha
          
          cat(sprintf("%s/%s at %s",
                      i0, NROW(PARAGRID_LINEARLS), Sys.time()), 
              "\n",
              "\n")
          
          tmpXgboost <- xgb.cv(
            data                  = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                                label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
            seed                  = 2020,
            nround                = NTREE,
            nthread               = CORE_SUB_CNT,
            early_stopping_rounds = EARLY_STOP_NROUND,
            nfold                 = 5,
            verbose               = 0,
            metrics               = "rmse",
            booster               = "gblinear",
            lambda                = tmpLambda,
            alpha                 = tmpAlpha
          )
          
          # ## Train vs Valid's RMSE Plot
          # tmpXboost_RMSE_Log <- tmpXgboost$evaluation_log %>%
          #   dplyr::select(iter, train_rmse_mean, test_rmse_mean) %>%
          #   reshape2::melt(id.vars = "iter") %>%
          #   rename(Iter = iter, Category = variable, RMSE = value)
          # 
          # tmpXboost_RMSE_Log %>%
          #   dplyr::filter(RMSE <= 3) %>%
          #   ggplot() +
          #   geom_point(aes(x = Iter, y = RMSE, group = Category, col = Category),
          #              alpha = 0.5, size = 0.1)
          
          BEST_ITER       <- tmpXgboost$best_iteration
          ## Train
          TRAIN_RMSE_MEAN <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$train_rmse_mean
          TRAIN_RMSE_SD   <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$train_rmse_std
          ## Valid
          VALID_RMSE_MEAN <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$test_rmse_mean
          VALID_RMSE_SD   <- tmpXgboost$evaluation_log %>% dplyr::filter(iter == BEST_ITER) %>% .$test_rmse_std
          
          ## Result
          tmpRMSE_Result <- data.table(
            LAMBDA                = tmpLambda,
            ALPHA                 = tmpAlpha,
            BEST_ITER             = BEST_ITER,
            TRAIN_RMSE_MEAN       = TRAIN_RMSE_MEAN,
            TRAIN_RMSE_SD         = TRAIN_RMSE_SD,
            VALID_RMSE_MEAN       = VALID_RMSE_MEAN,
            VALID_RMSE_SD         = VALID_RMSE_SD
          )
          
          return(tmpRMSE_Result)
        }
        
        if ( CORE_N > 1 ){
          stopCluster(MAIN_CLUSTER)
        }
        
        paraResult_Linear <- paraResult_Linear %>% dplyr::arrange(VALID_RMSE_MEAN, VALID_RMSE_SD)
        
        bestTunePara_Linear <- paraResult_Linear %>% 
          dplyr::filter(VALID_RMSE_MEAN == min(VALID_RMSE_MEAN)) %>% 
          dplyr::arrange(VALID_RMSE_MEAN, VALID_RMSE_SD) %>% .[1, ]
        
        LAMBDA            <- bestTunePara_Linear$LAMBDA
        ALPHA             <- bestTunePara_Linear$ALPHA
        BEST_ITER         <- round(bestTunePara_Linear$BEST_ITER, -2)
        
        
        Fit_XGB_Linear <- xgb.train(
          data                  = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                              label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
          seed                  = 2020,
          nround                = BEST_ITER,
          nthread               = CORE_N,
          early_stopping_rounds = EARLY_STOP_NROUND,
          verbose               = 0, 
          watchlist             = list(
            train = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
            test = xgb.DMatrix(data  = data.matrix(testData %>% dplyr::select_(.dots = VAR_LS)),
                               label = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
          ),
          booster               = "gblinear",
          lambda                = LAMBDA,
          alpha                 = ALPHA
        )
        
      } else {
        
        ## Do not Grid Search 
        Fit_XGB_Linear <- xgb.train(
          data                  = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                              label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
          seed                  = 2020,
          nround                = NTREE,
          nthread               = CORE_N,
          early_stopping_rounds = EARLY_STOP_NROUND,
          verbose               = 0, 
          watchlist             = list(
            train = xgb.DMatrix(data  = data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)),
                                label = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull),
            test = xgb.DMatrix(data  = data.matrix(testData %>% dplyr::select_(.dots = VAR_LS)),
                               label = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull)
          ),
          booster               = "gblinear",
          lambda                = PARAGRID_LINEARLS$lambda,
          alpha                 = PARAGRID_LINEARLS$alpha
          
        )
        
      } ## End of XGB_Linear
      ########################################################
      
      
      # Fit_XGB_Linear$evaluation_log %>%
      #   dplyr::select(iter, train_rmse, test_rmse) %>%
      #   melt(id.vars = "iter") %>%
      #   setnames(c("ITER", "CATEGORY", "VALUE")) %>%
      #   .[VALUE < 3] %>%
      #   ggplot() +
      #   geom_point(aes(x = ITER, y = VALUE, col = CATEGORY, fill = CATEGORY))
      
      
      #### .... Predict ####
      Pred_XGB_Linear <- predict(Fit_XGB_Linear,
                                 newdata = data.matrix(testData %>% dplyr::select_(.dots = VAR_LS)))
      Pred_XGB_Linear <- Pred_XGB_Linear %>% sapply(function(x) { max(min(x, PRED_MAX), PRED_MIN) })
      
      RMSE_XGB_Linear <- RMSE(pred = Pred_XGB_Linear,
                              obs  = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull())
      MAE_XGB_Linear  <- MAE(pred = Pred_XGB_Linear,
                             obs  = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull())
      
      #### .... Variable Importance ####
      VariableImportance_XGB_Linear <- xgb.importance(model = Fit_XGB_Linear) %>% 
        as.data.frame() %>% 
        select(Feature, Weight) %>% 
        setnames(c("VAR_NAME", "VAR_IMPORT")) %>% 
        dplyr::mutate(VAR_IMPORT = abs(VAR_IMPORT) / max(VAR_IMPORT)) %>% 
        dplyr::mutate(VISCOSITY  = VISCOSITY_TARGET,
                      TEST_SET   = TEST_SET,
                      VAR_COMB   = VAR_COMB,
                      METHOD     = "XGB_Linear") %>% 
        dplyr::select(VISCOSITY, TEST_SET, VAR_COMB, METHOD, VAR_NAME, VAR_IMPORT, everything())
      
      VariableImportance_XGB_Linear <- data.table(VAR_NAME = VAR_LS) %>% 
        merge(VariableImportance_XGB_Linear,
              by = "VAR_NAME",
              all.x = TRUE) %>% 
        replace_na(list(VISCOSITY  = VISCOSITY_TARGET,
                        TEST_SET   = TEST_SET,
                        VAR_COMB   = VAR_COMB,
                        METHOD     = "XGB_Linear",
                        VAR_IMPORT = 0)) %>% 
        .[order(desc(VAR_IMPORT))] %>% 
        dplyr::select(VISCOSITY, TEST_SET, VAR_COMB, METHOD, VAR_NAME, VAR_IMPORT, everything())
      
      
      
      
      
      #### 7. Result Save ####
      #### .. RMSE, MAE Result ####
      tmpERROR_Result <- data.table(
        VISCOSITY       = VISCOSITY_TARGET,
        TEST_SET        = TEST_SET,
        VAR_COMB        = VAR_COMB,
        RMSE_LM         = RMSE_LM,
        RMSE_RF         = RMSE_RF,
        RMSE_XGB        = RMSE_XGB,
        RMSE_XGB_LINEAR = RMSE_XGB_Linear,
        MAE_LM          = MAE_LM,
        MAE_RF          = MAE_RF,
        MAE_XGB         = MAE_XGB,
        MAE_XGB_LINEAR  = MAE_XGB_Linear
      )
      
      ERROR_Result <- rbind(ERROR_Result, tmpERROR_Result)
      
      
      
      #### .. OBS vs Predict Plot ####
      #### .... TRAIN ####
      if ( VAR_COMB != "ALL" ){
        CompareTrain_DT <- data.table(
          IDX        = 1:NROW(trainData),
          OBS        = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull(),
          LM         = Fit_LM$finalModel$fitted.values,
          RF         = predict(Fit_RF, 
                               newdata = as.h2o(trainData %>% dplyr::select_(.dots = c(VAR_LS, TARGET_VAR)))
          ) %>% as.data.frame() %>% pull(),
          XGB        = predict(Fit_XGB, data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS))),
          XGB_LINEAR = predict(Fit_XGB_Linear, data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)))
        ) %>% melt(id.vars = c("IDX", "OBS")) %>% 
          setnames(c("IDX", "OBS", "CATEGORY", "VALUE"))
      } else {
        CompareTrain_DT <- data.table(
          IDX        = 1:NROW(trainData),
          OBS        = trainData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull(),
          LASSO      = predict(Fit_LM, newdata = trainData),
          RF         = predict(Fit_RF, 
                               newdata = as.h2o(trainData %>% dplyr::select_(.dots = c(VAR_LS, TARGET_VAR)))
          ) %>% as.data.frame() %>% pull(),
          XGB        = predict(Fit_XGB, data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS))),
          XGB_LINEAR = predict(Fit_XGB_Linear, data.matrix(trainData %>% dplyr::select_(.dots = VAR_LS)))
        ) %>% melt(id.vars = c("IDX", "OBS")) %>% 
          setnames(c("IDX", "OBS", "CATEGORY", "VALUE"))
      }
      CompareTrain_DT[, VALUE := sapply(VALUE, function(x) { max(min(x, PRED_MAX), PRED_MIN) })]
      
      
      OBS_Predict_Train_Plot <- CompareTrain_DT %>% 
        ggplot() +
        geom_point(aes(x = IDX, y = OBS),   col = "black") +
        geom_point(aes(x = IDX, y = VALUE, col = CATEGORY)) +
        labs(title = sprintf("OBS vs Predict Plot \n %s \n Test:%s / Variable Comb:%s", 
                             VISCOSITY_TARGET,
                             TEST_SET, VAR_COMB),
             x = "IDX", y = "VALUE") +
        theme(plot.title = element_text(hjust = 0.5, size = 15),
              legend.position = "bottom") +
        facet_wrap(~CATEGORY,
                   nrow = 2
                   , scale = "free"
        )
      
      
      ## Save
      ggsave(file   = sprintf("%s/%s_Obs_Predict_Train_Plot_Test_%s_VarComb_%s__%s.png", 
                              SAVE_FOLD,
                              VISCOSITY_TARGET,
                              gsub("-", "", TEST_SET), VAR_COMB,
                              gsub("-", "", TODAY)),
             plot   = OBS_Predict_Train_Plot,
             width  = 45,
             height = 30,
             units  = "cm")
      
      #### .... TEST ####
      if ( VAR_COMB != "ALL" ){
        Compare_DT <- data.table(
          IDX        = 1:NROW(testData),
          OBS        = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull(),
          LM         = Pred_LM,
          RF         = Pred_RF,
          XGB        = Pred_XGB,
          XGB_LINEAR = Pred_XGB_Linear
        ) %>% melt(id.vars = c("IDX", "OBS")) %>% 
          setnames(c("IDX", "OBS", "CATEGORY", "VALUE"))
      } else {
        Compare_DT <- data.table(
          IDX        = 1:NROW(testData),
          OBS        = testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull(),
          LASSO      = Pred_LM,
          RF         = Pred_RF,
          XGB        = Pred_XGB,
          XGB_LINEAR = Pred_XGB_Linear
        ) %>% melt(id.vars = c("IDX", "OBS")) %>% 
          setnames(c("IDX", "OBS", "CATEGORY", "VALUE"))
      }
      
      OBS_Predict_Plot <- Compare_DT %>% 
        ggplot() +
        geom_point(aes(x = IDX, y = OBS),   col = "black") +
        geom_point(aes(x = IDX, y = VALUE, col = CATEGORY)) +
        labs(title = sprintf("OBS vs Predict Plot \n %s \n Test:%s / Variable Comb:%s", 
                             VISCOSITY_TARGET,
                             TEST_SET, VAR_COMB),
             x = "IDX", y = "VALUE") +
        theme(plot.title = element_text(hjust = 0.5, size = 15),
              legend.position = "bottom") +
        facet_wrap(~CATEGORY,
                   nrow = 2
                   , scale = "free"
        )
      
      
      ## Save
      ggsave(file   = sprintf("%s/%s_Obs_Predict_Plot_Test_%s_VarComb_%s__%s.png", 
                              SAVE_FOLD,
                              VISCOSITY_TARGET,
                              gsub("-", "", TEST_SET), VAR_COMB,
                              gsub("-", "", TODAY)),
             plot   = OBS_Predict_Plot,
             width  = 45,
             height = 30,
             units  = "cm")
      
      
      #### .. Residual Plot ####
      if ( VAR_COMB != "ALL" ){
        Compare_Error <- data.table(
          IDX        = 1:NROW(testData),
          ERROR_LM   = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_LM,
          ERROR_RF   = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_RF,
          ERROR_XGB  = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_XGB,
          ERROR_XGB_LINEAR = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_XGB_Linear
        ) %>% melt(id.vars = c("IDX")) %>% 
          setnames(c("IDX", "CATEGORY", "VALUE"))
      } else {
        Compare_Error <- data.table(
          IDX          = 1:NROW(testData),
          ERROR_LASSSO = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_LM,
          ERROR_RF     = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_RF,
          ERROR_XGB    = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_XGB,
          ERROR_XGB_LINEAR = (testData %>% dplyr::select_(.dots = TARGET_VAR) %>% pull()) - Pred_XGB_Linear
        ) %>% melt(id.vars = c("IDX")) %>% 
          setnames(c("IDX", "CATEGORY", "VALUE"))
      }
      
      Residual_Plot <- Compare_Error %>% 
        ggplot() +
        geom_point(aes(x = IDX, y = VALUE, col = CATEGORY)) +
        labs(title = sprintf("Residual(Error) Plot \n %s \n Test:%s / Variable Comb:%s", 
                             VISCOSITY_TARGET,
                             TEST_SET, VAR_COMB),
             x = "TIME", y = "VALUE") +
        theme(plot.title = element_text(hjust = 0.5, size = 15),
              legend.position = "bottom") +
        geom_hline(yintercept = 0, col = "black") +
        facet_wrap(~CATEGORY,
                   nrow = 2
                   # , scale = "free"
        )
      
      ## Save
      ggsave(file   = sprintf("%s/%s_Residual_Plot_Test_%s_VarComb_%s__%s.png", 
                              SAVE_FOLD,
                              VISCOSITY_TARGET,
                              gsub("-", "", TEST_SET), VAR_COMB,
                              gsub("-", "", TODAY)),
             plot   = Residual_Plot,
             width  = 45,
             height = 30,
             units  = "cm")
      
      #### .. Variable Importance ####
      VAR_Import_Result <- VAR_Import_Result %>% 
        rbind(VariableImportance_RF) %>% 
        rbind(VariableImportance_XGB) %>% 
        rbind(VariableImportance_XGB_Linear)
      
      
    } ## Roof of Variable Combinations
    
  } ## Roof of Test Set
  
  
  #### .. SAVE Final Result ####
  if( vs0 == NROW(VISCOSITY_LS) ){ 
    
    write.csv(ERROR_Result,
              file = sprintf("%s/RMSE_MAE_Result_%s.csv", SAVE_FOLD, gsub("-", "", TODAY)),
              row.names = FALSE,
              fileEncoding = "euc-kr")
    write.csv(VAR_Import_Result,
              file = sprintf("%s/Var_Import_%s.csv",
                             SAVE_FOLD, gsub("-", "", TODAY)),
              row.names = FALSE,
              fileEncoding = "euc-kr")
    write.csv(GLM_Use_Var,
              file = sprintf("%s/GLM_USE_VARIABLE_%s.csv", 
                             SAVE_FOLD, gsub("-", "", TODAY)),
              row.names = FALSE,
              fileEncoding = "euc-kr")
    write.csv(baseLine_ERROR,
              file = sprintf("%s/BASELINE_ERROR_%s.csv", 
                             SAVE_FOLD, gsub("-", "", TODAY)),
              row.names = FALSE,
              fileEncoding = "euc-kr")
    
  }
  
  
} ## Roof of Viscosity
# h2o.shutdown()

# ERROR_Result[order(RMSE_LM)] %>% dplyr::select(TEST_SET, VAR_COMB, starts_with("RMSE"))
# ERROR_Result[order(RMSE_RF)] %>% dplyr::select(TEST_SET, VAR_COMB, starts_with("RMSE"))
# ERROR_Result[order(RMSE_XGB)] %>% dplyr::select(TEST_SET, VAR_COMB, starts_with("RMSE"))
# ERROR_Result[order(RMSE_XGB_LINEAR)] %>% dplyr::select(TEST_SET, VAR_COMB, starts_with("RMSE"))

TOTAL_END_TIME <- Sys.time()

cat("\n",
    sprintf("===== Total Process is END! ====="),
    '\n Start : ', paste0(TOTAL_ST_TIME),
    '\n End   : ', paste0(TOTAL_END_TIME),
    '\n',
    capture.output(difftime(TOTAL_END_TIME, TOTAL_ST_TIME, units = "mins")),
    "\n",
    "\n")


