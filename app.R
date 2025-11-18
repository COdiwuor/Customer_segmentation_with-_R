# app.R
# ------------------------------------------------------------
# Shiny: Customer Segmentation (K-means / GMM / DBSCAN) + Personas
# ------------------------------------------------------------

# ---- Packages ----
# pkgs <- c("shiny","bslib","tidyverse","DT","cluster","factoextra","mclust","dbscan")
# to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
# if (length(to_install)) install.packages(to_install, dependencies = TRUE)
# lapply(pkgs, library, character.only = TRUE)
library(shiny)
library(bslib)
library(tidyverse)
library(DT)
library(cluster)
library(factoextra)
library(mclust)
library(dbscan)

# ---- Demo data (Mall-like) ----
# demo_data <- function(n = 200) {
#   set.seed(7)
#   tibble(
#     CustomerID     = seq_len(n),
#     Gender         = sample(c("Male","Female"), n, TRUE, c(0.45, 0.55)),
#     Age            = pmax(15, round(rnorm(n, 35, 12))),
#     `Annual Income (k$)` = pmax(10, round(rnorm(n, 60, 25))),
#     `Spending Score (1-100)` = pmin(100, pmax(1, round(rnorm(n, 55, 25))))
#   )
# }

# ---- Helpers ----
guess_gender_col <- function(df) {
  nm <- tolower(names(df)); idx <- which(nm %in% c("gender","sex"))
  if (length(idx)) names(df)[idx[1]] else NULL
}
scale_df <- function(df) as_tibble(scale(df), .name_repair = "minimal")

silhouette_for_k <- function(X, cl) {
  d <- dist(X, method = "euclidean")
  s <- cluster::silhouette(as.integer(cl), d)
  mean(s[, 3])
}

auto_kmeans <- function(X, k_min = 2, k_max = 10, nstart = 50, iter.max = 100) {
  ks <- k_min:k_max
  sils <- numeric(length(ks))
  models <- vector("list", length(ks))
  for (i in seq_along(ks)) {
    set.seed(42)
    fit <- kmeans(X, centers = ks[i], nstart = nstart, iter.max = iter.max, algorithm = "Lloyd")
    models[[i]] <- fit
    sils[i] <- silhouette_for_k(X, fit$cluster)
  }
  best_idx <- which.max(sils)
  list(model = models[[best_idx]], k = ks[best_idx], ks = ks, sils = sils)
}

# --- Persona labelling: data-driven, robust to label permutation --------------
# We classify each cluster on tertiles for Income/Spending and Age.
# Then map common combinations to clear persona names; provide sensible fallbacks.
label_personas <- function(profile_tbl) {
  # profile_tbl: cluster, age_mean, inc_mean, spend_mean, n
  cut3 <- function(x) {
    q <- quantile(x, probs = c(.33,.67), na.rm = TRUE, type = 7)
    dplyr::case_when(
      x <= q[1] ~ "low",
      x <= q[2] ~ "mid",
      TRUE      ~ "high"
    )
  }
  prof <- profile_tbl %>%
    mutate(
      age_cat   = cut3(age_mean),
      inc_cat   = cut3(inc_mean),
      spend_cat = cut3(spend_mean)
    )
  
  map_row <- function(a,i,s) {
    # Primary rules by income/spend
    if (i=="high" && s=="high")   return("Premium Big Spenders")
    if (i=="low"  && s=="high")   return("Trend Seekers")
    if (i=="high" && s=="low")    return("Frugal Professionals")
    if (i=="low"  && s=="low")    return("Budget Buyers")
    # Mid combos — use age to split
    if (s=="mid" && i %in% c("mid","high") && a %in% c("low","mid"))
      return("Selective Young Adults")
    if (s=="mid" && i %in% c("mid","low") && a=="high")
      return("Steady Shoppers")
    # Fallbacks:
    if (s=="high") return("Active Spenders")
    if (s=="low")  return("Conservative Buyers")
    return("Mixed Segment")
  }
  
  prof$persona <- purrr::pmap_chr(
    prof[,c("age_cat","inc_cat","spend_cat")],
    ~ map_row(..1, ..2, ..3)
  )
  prof %>% select(cluster, persona, age_cat, inc_cat, spend_cat)
}

# ---- UI ----
ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "flatly"),
  titlePanel("Customer Segmentation – Shiny"),
  sidebarLayout(
    sidebarPanel(
      h4("1) Data"),
      fileInput("file", "Upload CSV (optional)", accept = c(".csv")),
      helpText("If no file is uploaded, a demo Mall-like dataset is used."),
      uiOutput("feature_picker"),
      checkboxInput("do_scale", "Scale numeric features", TRUE),
      hr(),
      h4("2) Algorithm"),
      radioButtons("algo", NULL,
                   choices = c("K-means" = "kmeans",
                               "Gaussian Mixture (GMM)" = "gmm",
                               "DBSCAN" = "dbscan"),
                   selected = "kmeans"),
      conditionalPanel(
        condition = "input.algo == 'kmeans'",
        checkboxInput("auto_k", "Auto-select K by silhouette", TRUE),
        conditionalPanel(
          condition = "!input.auto_k",
          sliderInput("k", "K (clusters)", min = 2, max = 12, value = 6, step = 1)
        ),
        conditionalPanel(
          condition = "input.auto_k",
          sliderInput("k_range", "K search range", min = 2, max = 15, value = c(2, 10), step = 1)
        )
      ),
      conditionalPanel(
        condition = "input.algo == 'gmm'",
        sliderInput("g_max", "Max components (G)", min = 2, max = 15, value = 8, step = 1),
        helpText("GMM chooses G automatically by BIC up to this limit.")
      ),
      conditionalPanel(
        condition = "input.algo == 'dbscan'",
        sliderInput("eps", "eps (neighborhood radius)", min = 0.1, max = 3, value = 0.8, step = 0.05),
        numericInput("minPts", "minPts", value = 5, min = 2, max = 50, step = 1),
        helpText("Use kNN-distance plot tab to tune eps.")
      ),
      hr(),
      h4("3) Plots"),
      uiOutput("axis_picker"),
      actionButton("run", "Run Segmentation", class = "btn btn-primary")
    ),
    mainPanel(
      tabsetPanel(id = "tabs", type = "pills",
                  tabPanel("Data Summary",
                           br(),
                           fluidRow(
                             column(6, tableOutput("dim_info")),
                             column(6, tableOutput("gender_info"))
                           ),
                           DTOutput("head_table")
                  ),
                  tabPanel("Cluster Plot",
                           br(),
                           plotOutput("feature_plot", height = 420)
                  ),
                  tabPanel("PCA Plot",
                           br(),
                           plotOutput("pca_plot", height = 420)
                  ),
                  tabPanel("Diagnostics",
                           br(),
                           conditionalPanel("input.algo == 'kmeans'",
                                            plotOutput("sil_plot", height = 350),
                                            verbatimTextOutput("k_choice")
                           ),
                           conditionalPanel("input.algo != 'kmeans'",
                                            plotOutput("sil_plot_other", height = 350)
                           ),
                           conditionalPanel("input.algo == 'dbscan'",
                                            plotOutput("knn_plot", height = 300)
                           )
                  ),
                  tabPanel("Profiles",
                           br(),
                           h4("Numeric summary by cluster"),
                           DTOutput("profile_num"),
                           br(),
                           h4("Gender mix by cluster"),
                           DTOutput("profile_gender")
                  ),
                  tabPanel("Assignments",
                           br(),
                           downloadButton("download_csv", "Download segmented CSV"),
                           br(), br(),
                           DTOutput("assign_table")
                  )
      )
    )
  )
)

# ---- Server ----
server <- function(input, output, session) {
  
  # Data
  raw_data <- reactive({
    req(input$file)  # This waits until a file is uploaded
    readr::read_csv(input$file$datapath, show_col_types = FALSE)
    
    #old code
    #if (is.null(input$file)) 
      #print("Please select a csv file")
      #demo_data()
    #else readr::read_csv(input$file$datapath, show_col_types = FALSE)
  })
  
  # Feature picker
  output$feature_picker <- renderUI({
    df <- raw_data()
    num_cols <- names(df)[sapply(df, is.numeric)]
    validate(need(length(num_cols) >= 2, "Need at least two numeric columns."))
    checkboxGroupInput("features","Select numeric features",
                       choices = num_cols, selected = num_cols)
  })
  
  # Axes picker
  output$axis_picker <- renderUI({
    req(input$features)
    fluidRow(
      column(6, selectInput("xvar","X-axis", choices = input$features, selected = input$features[1])),
      column(6, selectInput("yvar","Y-axis", choices = input$features, selected = input$features[min(2,length(input$features))]))
    )
  })
  
  # Summary tables
  output$dim_info <- renderTable({
    df <- raw_data()
    tibble(Metric = c("Rows","Columns"), Value = c(nrow(df), ncol(df)))
  })
  output$gender_info <- renderTable({
    df <- raw_data(); gcol <- guess_gender_col(df)
    if (is.null(gcol)) return(tibble(Note = "No gender column detected."))
    df %>% count(.data[[gcol]], name = "n") %>%
      mutate(pct = round(100*n/sum(n),1)) %>% rename(Gender = !!gcol)
  })
  output$head_table <- renderDT({
    datatable(head(raw_data(), 10), options = list(scrollX = TRUE))
  })
  
  # Core computation
  result <- eventReactive(input$run, {
    df <- raw_data(); req(input$features)
    X  <- df %>% select(all_of(input$features))
    validate(need(ncol(X) >= 2, "Select at least two numeric features."))
    Xs <- if (isTRUE(input$do_scale)) scale_df(X) else as_tibble(X)
    
    algo <- input$algo; clusters <- NULL; meta <- list()
    
    if (algo == "kmeans") {
      if (isTRUE(input$auto_k)) {
        kr <- auto_kmeans(Xs, k_min = input$k_range[1], k_max = input$k_range[2])
        fit <- kr$model; clusters <- factor(fit$cluster)
        meta$k <- kr$k; meta$ks <- kr$ks; meta$sils <- kr$sils
        meta$avg_sil <- silhouette_for_k(Xs, clusters)
        meta$algo_label <- paste0("K-means (auto-K=", meta$k, ")")
      } else {
        set.seed(42)
        fit <- kmeans(Xs, centers = input$k, nstart = 50, iter.max = 200, algorithm = "Lloyd")
        clusters <- factor(fit$cluster)
        meta$k <- input$k
        meta$avg_sil <- silhouette_for_k(Xs, clusters)
        meta$algo_label <- paste0("K-means (K=", meta$k, ")")
      }
    }
    
    if (algo == "gmm") {
      set.seed(42)
      gfit <- mclust::Mclust(as.matrix(Xs), G = 1:input$g_max)
      clusters <- factor(gfit$classification)
      meta$g <- gfit$G
      meta$avg_sil <- silhouette_for_k(Xs, clusters)
      meta$algo_label <- paste0("GMM (G=", meta$g, ")")
      meta$gmm <- gfit
    }
    
    if (algo == "dbscan") {
      db <- dbscan::dbscan(as.matrix(Xs), eps = input$eps, minPts = input$minPts)
      clusters <- factor(db$cluster) # 0 = noise
      meta$algo_label <- paste0("DBSCAN (eps=", input$eps, ", minPts=", input$minPts, ")")
      non_noise <- which(db$cluster > 0)
      meta$avg_sil <- if (length(unique(db$cluster[non_noise])) >= 2) {
        silhouette_for_k(Xs[non_noise, ], db$cluster[non_noise])
      } else NA_real_
      meta$db <- db
    }
    
    # ---- Build cluster profiles and assign personas (for ALL algos) ----------
    assigned <- bind_cols(df, cluster = clusters)
    # compute means per cluster (using only selected numeric features when present)
    means <- assigned %>%
      group_by(cluster) %>%
      summarise(
        n = n(),
        age_mean   = if ("Age" %in% names(df)) mean(Age, na.rm = TRUE) else NA_real_,
        inc_mean   = if ("Annual Income (k$)" %in% names(df)) mean(`Annual Income (k$)`, na.rm = TRUE) else NA_real_,
        spend_mean = if ("Spending Score (1-100)" %in% names(df)) mean(`Spending Score (1-100)`, na.rm = TRUE) else NA_real_,
        .groups = "drop"
      ) %>%
      mutate(cluster = as.character(cluster))
    
    # If any of the three core features missing, fall back to using available ones:
    if (all(is.na(means$age_mean)) && all(is.na(means$inc_mean)) && all(is.na(means$spend_mean))) {
      # compute generic averages across selected features
      tmp <- assigned %>%
        group_by(cluster) %>%
        summarise(n = n(), across(all_of(input$features), mean, .names = "feat_{.col}"), .groups = "drop") %>%
        mutate(cluster = as.character(cluster))
      means <- tmp %>% mutate(age_mean = NA_real_, inc_mean = NA_real_, spend_mean = NA_real_)
    }
    
    persona_map <- label_personas(
      means %>% transmute(
        cluster = cluster,
        age_mean = age_mean,
        inc_mean = inc_mean,
        spend_mean = spend_mean
      )
    )
    
    assigned <- assigned %>%
      mutate(cluster = as.character(cluster)) %>%
      left_join(persona_map %>% select(cluster, persona), by = "cluster") %>%
      mutate(cluster = factor(cluster),
             persona = factor(persona))
    
    meta$profiles_core <- means %>% left_join(persona_map, by = "cluster")
    list(df = df, X = X, Xs = Xs, assign = assigned, meta = meta)
  }, ignoreInit = TRUE)
  
  # Scatter by Persona
  output$feature_plot <- renderPlot({
    r <- result(); req(r, input$xvar, input$yvar)
    ggplot(r$assign, aes(.data[[input$xvar]], .data[[input$yvar]], color = persona)) +
      geom_point(alpha = 0.85) +
      labs(
        title = paste("Clusters by", input$xvar, "vs", input$yvar),
        subtitle = paste0(r$meta$algo_label,
                          if (!is.na(r$meta$avg_sil)) paste0(" | Avg silhouette: ", round(r$meta$avg_sil, 3)) else ""),
        color = "Persona"
      ) +
      theme_minimal()
  })
  
  # PCA by Persona
  output$pca_plot <- renderPlot({
    r <- result(); req(r)
    p <- prcomp(r$Xs, center = TRUE, scale. = FALSE)
    dfp <- as_tibble(p$x[, 1:2]) %>% set_names(c("PC1","PC2")) %>%
      bind_cols(persona = r$assign$persona)
    ggplot(dfp, aes(PC1, PC2, color = persona)) +
      geom_point(alpha = 0.9) +
      theme_minimal() +
      labs(title = "PCA projection by persona", subtitle = r$meta$algo_label, color = "Persona")
  })
  
  # Diagnostics
  output$sil_plot <- renderPlot({
    r <- result(); req(r); validate(need(grepl("^K-means", r$meta$algo_label), "Silhouette sweep shown for K-means only."))
    if (!isTRUE(input$auto_k)) {
      d <- dist(r$Xs, method = "euclidean")
      s <- silhouette(as.integer(r$assign$cluster), d)
      fviz_silhouette(s) + theme_minimal()
    } else {
      tibble(k = r$meta$ks, avg_sil = r$meta$sils) %>%
        ggplot(aes(k, avg_sil)) +
        geom_line() + geom_point() +
        geom_vline(xintercept = r$meta$k, linetype = 2) +
        labs(title = "Average silhouette vs K (auto selection)",
             subtitle = paste("Chosen K =", r$meta$k),
             x = "K", y = "Average silhouette") +
        theme_minimal()
    }
  })
  output$k_choice <- renderPrint({
    r <- result(); req(r); if (!grepl("^K-means", r$meta$algo_label)) return(invisible())
    cat("Final K-means choice:\n")
    if (isTRUE(input$auto_k)) cat("  K =", r$meta$k, " (auto)\n") else cat("  K =", r$meta$k, "\n")
    cat("Average silhouette:", round(r$meta$avg_sil, 4), "\n")
  })
  output$sil_plot_other <- renderPlot({
    r <- result(); req(r)
    if (grepl("^K-means", r$meta$algo_label)) return(invisible(NULL))
    if (grepl("^DBSCAN", r$meta$algo_label)) validate(need(FALSE, "Silhouette shown only for K-means/GMM."))
    d <- dist(r$Xs, method = "euclidean")
    s <- silhouette(as.integer(r$assign$cluster), d)
    fviz_silhouette(s) + theme_minimal()
  })
  output$knn_plot <- renderPlot({
    r <- result(); req(r); validate(need(grepl("^DBSCAN", r$meta$algo_label), "kNN distance plot used for DBSCAN only."))
    dbscan::kNNdistplot(as.matrix(r$Xs), k = input$minPts)
    abline(h = input$eps, lty = 2); title("kNN distance plot (horizontal line = eps)")
  })
  
  # Profiles (with Persona)
  output$profile_num <- renderDT({
    r <- result(); req(r)
    # Build numeric profile from assigned + selected features
    nums <- r$df %>% select(any_of(input$features))
    prof <- r$assign %>%
      group_by(cluster, persona) %>%
      summarise(
        n = n(),
        across(all_of(colnames(nums)),
               list(mean = ~mean(.x, na.rm = TRUE),
                    median = ~median(.x, na.rm = TRUE)),
               .names = "{.col}_{.fn}"),
        .groups = "drop"
      ) %>%
      arrange(persona)
    datatable(prof, options = list(scrollX = TRUE))
  })
  output$profile_gender <- renderDT({
    r <- result(); req(r)
    gcol <- guess_gender_col(r$assign)
    if (is.null(gcol)) return(datatable(tibble(Note = "No gender column detected.")))
    mix <- r$assign %>%
      count(persona, .data[[gcol]], name = "n") %>%
      group_by(persona) %>% mutate(pct = round(100*n/sum(n),1)) %>% ungroup() %>%
      rename(Gender = !!gcol) %>% arrange(persona, desc(pct))
    datatable(mix, options = list(scrollX = TRUE))
  })
  
  # Assignments + download (with Persona)
  output$assign_table <- renderDT({
    r <- result(); req(r)
    out <- r$assign %>% relocate(persona, .after = last_col())
    datatable(out, options = list(scrollX = TRUE))
  })
  output$download_csv <- downloadHandler(
    filename = function() paste0("segmented_customers_", Sys.Date(), ".csv"),
    content = function(file) {
      r <- result(); req(r)
      readr::write_csv(r$assign, file)
    }
  )
}

shinyApp(ui, server)
