/* --- 1. PREPROCESSING & MODEL CONFIGURATION --- */

// === TASK 1: REGRESSION (Car Price) CONFIG ===
const REGRESSION_CONFIG = {
    taskType: 'regression',
    models: {
        // Friendly Name: [filename, R2 score]
        "XGBoost (R²: 0.979)": ["XGBoost_CarPrice.onnx", 0.9788],
        "Linear_Plus_Non-Linear (R²: 0.960)": ["Linear_Plus_Non-Linear_CarPrice.onnx", 0.9599],
        "Deep_Net_(2_Hidden_Layers) (R²: 0.946)": ["Deep_Net_(2_Hidden_Layers)_CarPrice.onnx", 0.9457],
        "MLP_(1_Hidden_Layer) (R²: 0.924)": ["MLP_(1_Hidden_Layer)_CarPrice.onnx", 0.9242],
        "Linear_Regression (R²: 0.687)": ["Linear_Regression_CarPrice.onnx", 0.6867]
    },
    
    // --- PASTED FROM YOUR PYTHON SCRIPT ---
    "inputFeatureCount": 20,
    "limits": {
      "year": {
        "min": 1994.0,
        "max": 2020.0
      },
      "km_driven": {
        "min": 1.0,
        "max": 1500000.0
      },
      "mileage": {
        "min": 0.0,
        "max": 42.0
      },
      "engine": {
        "min": 624.0,
        "max": 3604.0
      },
      "max_power": {
        "min": 32.8,
        "max": 400.0
      },
      "seats": {
        "min": 2.0,
        "max": 10.0
      }
    },
    "means": {
      "year": 2013.9843453510437,
      "km_driven": 69277.96078431372,
      "mileage": 19.401921252371917,
      "engine": 1462.6310879190387,
      "max_power": 91.72465132827325,
      "seats": 5.423308032890575
    },
    "stds": {
      "year": 3.885526960960459,
      "km_driven": 51889.77839224393,
      "mileage": 4.07355869973661,
      "engine": 506.3969975132824,
      "max_power": 35.82674440365109,
      "seats": 0.9624352082278965
    },
    "categories": {
      "fuel": [
        "CNG",
        "Diesel",
        "LPG",
        "Petrol"
      ],
      "seller_type": [
        "Dealer",
        "Individual",
        "Trustmark Dealer"
      ],
      "transmission": [
        "Automatic",
        "Manual"
      ],
      "owner": [
        "First Owner",
        "Fourth & Above Owner",
        "Second Owner",
        "Test Drive Car",
        "Third Owner"
      ]
    },
    "output_scaler": {
      "mean": 650379.954459203,
      "std": 808371.080965429
    },
    "binaryFeatures": []
    // --- END OF REGRESSION PASTE ---
};

// === TASK 2: CLASSIFICATION (Insurance Claim) CONFIG ===
const CLASSIFICATION_CONFIG = {
    taskType: 'classification',
    models: {
        "MLP_Net (F1: 0.905)": ["MLP_Net_Claim.onnx", 0.9051],
        "DL_Net (F1: 0.905)": ["DL_Net_Claim.onnx", 0.9051]
    },

    // --- PASTED FROM YOUR PYTHON SCRIPT ---
    "inputFeatureCount": 98,
    "limits": {
      "policy_tenure": { "min": 0.002735272840513, "max": 1.39664107699389 },
      "age_of_car": { "min": 0.0, "max": 1.0 },
      "age_of_policyholder": { "min": 0.288461538461538, "max": 1.0 },
      "population_density": { "min": 290.0, "max": 73430.0 },
      "make": { "min": 1.0, "max": 5.0 },
      "airbags": { "min": 1.0, "max": 6.0 },
      "displacement": { "min": 796.0, "max": 1498.0 },
      "cylinder": { "min": 3.0, "max": 4.0 },
      "gear_box": { "min": 5.0, "max": 6.0 },
      "turning_radius": { "min": 4.5, "max": 5.2 },
      "length": { "min": 3445.0, "max": 4300.0 },
      "width": { "min": 1475.0, "max": 1811.0 },
      "height": { "min": 1475.0, "max": 1825.0 },
      "gross_weight": { "min": 1051.0, "max": 1720.0 },
      "ncap_rating": { "min": 0.0, "max": 5.0 }
    },
    "means": {
      "policy_tenure": 0.6124082762984717,
      "age_of_car": 0.06934333198216458,
      "age_of_policyholder": 0.46969737375461346,
      "population_density": 18829.491946323044,
      "make": 1.7667740490260917,
      "airbags": 3.1378618821069697,
      "displacement": 1163.1433447827108,
      "cylinder": 3.62827213961129,
      "gear_box": 5.2455358095278735,
      "turning_radius": 4.853295073923154,
      "length": 3851.10814328078,
      "width": 1672.4819832312846,
      "height": 1553.4532033366756,
      "gross_weight": 1385.409788150961,
      "ncap_rating": 1.7637872549228768
    },
    "stds": {
      "policy_tenure": 0.4147342105159749,
      "age_of_car": 0.05638831974440441,
      "age_of_policyholder": 0.1227980232397579,
      "population_density": 17660.783028029255,
      "make": 1.1397962126604073,
      "airbags": 1.8332854784878638,
      "displacement": 266.06608390902863,
      "cylinder": 0.4832662394576531,
      "gear_box": 0.4304044327924204,
      "turning_radius": 0.22792782380479665,
      "length": 311.07572871062575,
      "width": 111.98444121120737,
      "height": 79.6717763875999,
      "gross_weight": 212.47848046637978,
      "ncap_rating": 1.3901636745579966
    },
    "categories": {
      "area_cluster": [
        "C1", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
        "C18", "C19", "C2", "C20", "C21", "C22", "C3", "C4", "C5",
        "C6", "C7", "C8", "C9"
      ],
      "segment": [ "A", "B1", "B2", "C1", "C2", "Utility" ],
      "model": [ "M1", "M10", "M11", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9" ],
      "fuel_type": [ "CNG", "Diesel", "Petrol" ],
      "is_esc": [ "No", "Yes" ],
      "is_adjustable_steering": [ "No", "Yes" ],
      "is_tpms": [ "No", "Yes" ],
      "is_parking_sensors": [ "No", "Yes" ],
      "is_parking_camera": [ "No", "Yes" ],
      "rear_brakes_type": [ "Disc", "Drum" ],
      "transmission_type": [ "Automatic", "Manual" ],
      "steering_type": [ "Electric", "Manual", "Power" ],
      "is_front_fog_lights": [ "No", "Yes" ],
      "is_rear_window_wiper": [ "No", "Yes" ],
      "is_rear_window_washer": [ "No", "Yes" ],
      "is_rear_window_defogger": [ "No", "Yes" ],
      "is_brake_assist": [ "No", "Yes" ],
      "is_power_door_locks": [ "No", "Yes" ],
      "is_central_locking": [ "No", "Yes" ],
      "is_power_steering": [ "No", "Yes" ],
      "is_driver_seat_height_adjustable": [ "No", "Yes" ],
      "is_day_night_rear_view_mirror": [ "No", "Yes" ],
      "is_ecw": [ "No", "Yes" ],
      "is_speed_alert": [ "No", "Yes" ]
    },
    "binaryFeatures": []
    // --- END OF CLASSIFICATION PASTE ---
};
// --- END OF CONFIGURATION ---


// --- Dictionary to hold loaded model sessions ---
const modelSessions = {
    regression: {},
    classification: {}
};

// --- DOM Elements ---
const ALL_DOM_ELEMENTS = {
    regression: {
        tabButton: document.getElementById('tab-btn-regression'),
        taskContainer: document.getElementById('task-regression'),
        form: document.getElementById('regression-form'),
        numericGrid: document.getElementById('reg-numeric-grid'),
        categoricalGrid: document.getElementById('reg-categorical-grid'),
        modelSelect: document.getElementById('reg-model-select'),
        predictButton: document.getElementById('reg-predict-button'),
        loadingSpinner: document.getElementById('reg-loading-spinner'),
        resultBox: document.getElementById('reg-result').querySelector('p')
    },
    classification: {
        tabButton: document.getElementById('tab-btn-classification'),
        taskContainer: document.getElementById('task-classification'),
        form: document.getElementById('classification-form'),
        numericGrid: document.getElementById('class-numeric-grid'),
        categoricalGrid: document.getElementById('class-categorical-grid'),
        modelSelect: document.getElementById('class-model-select'),
        predictButton: document.getElementById('class-predict-button'),
        loadingSpinner: document.getElementById('class-loading-spinner'),
        resultBox: document.getElementById('class-result').querySelector('p')
    }
};

/* --- 2. Main Application Logic --- */

/**
 * Main function to load ALL models and set up event listeners.
 */
async function main() {
    // --- Setup Tabs ---
    ALL_DOM_ELEMENTS.regression.tabButton.addEventListener('click', () => switchTab('regression'));
    ALL_DOM_ELEMENTS.classification.tabButton.addEventListener('click', () => switchTab('classification'));

    // --- Setup Forms ---
    ALL_DOM_ELEMENTS.regression.form.addEventListener('submit', (e) => {
        e.preventDefault();
        runPrediction(REGRESSION_CONFIG, ALL_DOM_ELEMENTS.regression);
    });
    ALL_DOM_ELEMENTS.classification.form.addEventListener('submit', (e) => {
        e.preventDefault();
        runPrediction(CLASSIFICATION_CONFIG, ALL_DOM_ELEMENTS.classification);
    });

    // --- Load Models and Populate Forms ---
    try {
        setLoading(ALL_DOM_ELEMENTS.regression, true, "");
        setLoading(ALL_DOM_ELEMENTS.classification, true, "");

        await Promise.all([
            loadModelsAndForms(REGRESSION_CONFIG, ALL_DOM_ELEMENTS.regression),
            loadModelsAndForms(CLASSIFICATION_CONFIG, ALL_DOM_ELEMENTS.classification)
        ]);
        
        setLoading(ALL_DOM_ELEMENTS.regression, false);
        setLoading(ALL_DOM_ELEMENTS.classification, false);
        
        console.log("All models loaded and forms populated.");
    } catch (e) {
        console.error("Critical error during initialization:", e);
        alert("Error loading models. Check console for details.");
    }
}

/**
 * Switches the active tab and task container.
 */
function switchTab(taskType) {
    const active = ALL_DOM_ELEMENTS[taskType];
    const inactiveTaskType = taskType === 'regression' ? 'classification' : 'regression';
    const inactive = ALL_DOM_ELEMENTS[inactiveTaskType];

    active.tabButton.classList.add('active');
    active.taskContainer.classList.remove('hidden');
    
    inactive.tabButton.classList.remove('active');
    inactive.taskContainer.classList.add('hidden');
}

/**
 * Loads models and populates the form for a specific task.
 */
async function loadModelsAndForms(config, dom) {
    console.log(`Loading models for ${config.taskType}...`);
    dom.predictButton.disabled = true;

    if (!config.limits || !config.means) {
        console.error(`Configuration data for ${config.taskType} is missing.`);
        dom.resultBox.textContent = `Error: Config data missing.`;
        dom.resultBox.className = 'claim';
        return; 
    }
    
    // 1. Load all models for this task
    const sessionHolder = modelSessions[config.taskType];
    for (const [name, [file, score]] of Object.entries(config.models)) {
        try {
            sessionHolder[name] = await ort.InferenceSession.create(file);
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            dom.modelSelect.appendChild(option);
            console.log(`Loaded ${file}`);
        } catch (e) {
            console.error(`Failed to load model ${file}:`, e);
            const option = document.createElement('option');
            option.value = name;
            option.textContent = `${name} (Failed to load)`;
            option.disabled = true;
            dom.modelSelect.appendChild(option);
        }
    }
    
    // 2. Populate numeric inputs
    for (const [feature, limits] of Object.entries(config.limits)) {
        // --- NEW: Create a container for the field ---
        const fieldContainer = document.createElement('div');
        fieldContainer.className = 'form-field-container';

        const label = document.createElement('label');
        label.htmlFor = `${config.taskType}-${feature}`;
        label.textContent = `${feature}:`;

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `${config.taskType}-${feature}`;
        input.min = limits.min;
        input.max = limits.max;
        
        if (limits.max - limits.min <= 2) input.step = 'any';
        else if (Number.isInteger(limits.min) && Number.isInteger(limits.max)) input.step = '1';
        else input.step = 'any';
        
        let defaultValue = config.means[feature];
        defaultValue = Math.max(limits.min, Math.min(limits.max, defaultValue));
        input.value = (input.step === '1') ? Math.round(defaultValue) : defaultValue.toFixed(4);
        
        // --- NEW: Create the limits label ---
        const limitsLabel = document.createElement('span');
        limitsLabel.className = 'input-limits';
        // Format numbers nicely
        let minText = Number(limits.min.toFixed(2));
        let maxText = Number(limits.max.toFixed(2));
        limitsLabel.textContent = `(Range: ${minText} to ${maxText})`;
        
        // --- NEW: Add all elements to the container ---
        fieldContainer.appendChild(label);
        fieldContainer.appendChild(input);
        fieldContainer.appendChild(limitsLabel); // Add the new label
        
        // Add the container to the grid
        dom.numericGrid.appendChild(fieldContainer);
    }
    
    // 3. Populate categorical inputs
    for (const [feature, categories] of Object.entries(config.categories)) {
        // --- NEW: Create a container for the field ---
        const fieldContainer = document.createElement('div');
        fieldContainer.className = 'form-field-container';
        
        const label = document.createElement('label');
        label.htmlFor = `${config.taskType}-${feature}`;
        label.textContent = `${feature}:`;

        const select = document.createElement('select');
        select.id = `${config.taskType}-${feature}`;
        
        for (const category of categories) {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            select.appendChild(option);
        }
        
        // --- NEW: Add elements to container ---
        fieldContainer.appendChild(label);
        fieldContainer.appendChild(select);
        
        // Add the container to the grid
        dom.categoricalGrid.appendChild(fieldContainer);
    }
    
    dom.predictButton.disabled = false;
    console.log(`Form populated for ${config.taskType}.`);
}

/**
 * Runs the prediction for the given task.
 */
async function runPrediction(config, dom) {
    setLoading(dom, true, "Running prediction...");
    
    try {
        const selectedModelName = dom.modelSelect.value;
        const session = modelSessions[config.taskType][selectedModelName];
        if (!session) throw new Error(`Model session for "${selectedModelName}" not found.`);

        const inputs = getFormValues(config, dom);
        const inputData = preprocess(inputs, config);
        const inputTensor = new ort.Tensor('float32', Float32Array.from(inputData), [1, config.inputFeatureCount]);
        
        let inputName = 'input1'; // Default for PyTorch
        let outputName = 'output1'; // Default for PyTorch
        
        if (selectedModelName.startsWith('XGBoost')) {
            inputName = 'float_input';
            if (config.taskType === 'regression') {
                outputName = 'variable'; // Default for XGBoost Regressor
            } else {
                outputName = 'output_label'; // Default for XGBoost Classifier
            }
        }
        
        const feeds = { [inputName]: inputTensor }; 
        const results = await session.run(feeds);
        
        if (!results[outputName]) {
            console.error("Output name mismatch!", "Expected:", outputName, "Got:", results);
            throw new Error(`Model output name mismatch. Expected '${outputName}'.`);
        }
        
        const outputData = results[outputName].data; 
        
        displayResult(outputData, config, dom, selectedModelName);
        
    } catch (e) {
        console.error(`Error during prediction for ${config.taskType}:`, e);
        dom.resultBox.textContent = `Error: ${e.message}`;
        dom.resultBox.className = 'claim'; 
    } finally {
        setLoading(dom, false);
    }
}

/**
 * Gathers all values from the form for a specific task.
 */
function getFormValues(config, dom) {
    const values = {};
    for (const feature of Object.keys(config.means)) {
        values[feature] = parseFloat(document.getElementById(`${config.taskType}-${feature}`).value);
    }
    for (const feature of Object.keys(config.categories)) {
        values[feature] = document.getElementById(`${config.taskType}-${feature}`).value;
    }
    return values;
}

/**
 * Preprocesses raw form data into a single flat array for the ONNX model.
 */
function preprocess(inputs, config) {
    const processedData = [];

    // 1. Scale Numeric Features
    for (const feature of Object.keys(config.means)) {
        const mean = config.means[feature];
        const std = config.stds[feature];
        const scaledValue = (inputs[feature] - mean) / std;
        processedData.push(scaledValue);
    }

    // 2. One-Hot Encode Categorical Features
    for (const [feature, categories] of Object.entries(config.categories)) {
        const selectedValue = inputs[feature];
        for (const category of categories) {
            processedData.push(category === selectedValue ? 1 : 0);
        }
    }

    // 3. Add Binary (Passthrough) Features (if any)
    for (const feature of config.binaryFeatures) {
        processedData.push(inputs[feature]);
    }

    // Final check
    if (processedData.length !== config.inputFeatureCount) {
        console.error(`Feature mismatch for ${config.taskType}: Expected ${config.inputFeatureCount}, got ${processedData.length}`);
        throw new Error(`Feature mismatch: Expected ${config.inputFeatureCount} features, but got ${processedData.length}`);
    }
    return processedData;
}

/**
 * Displays the final formatted prediction.
 */
function displayResult(outputData, config, dom, selectedModelName) {
    if (config.taskType === 'regression') {
        
        let finalPrice;
        if (selectedModelName.startsWith('XGBoost')) {
            // XGBoost output is already in Rupees. No de-scaling needed.
            finalPrice = outputData[0];
        } else {
            // PyTorch models output a scaled value. We must de-scale it.
            let scaledPrediction = outputData[0];
            let mean = config.output_scaler.mean;
            let std = config.output_scaler.std;
            finalPrice = (scaledPrediction * std) + mean;
        }
        
        dom.resultBox.textContent = `₹ ${finalPrice.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
        dom.resultBox.className = 'regression';
        
    } else if (config.taskType === 'classification') {
        
        let prediction;
        if (selectedModelName.startsWith('XGBoost')) {
            // XGBoost output is just the label, e.g., [1]
            prediction = outputData[0];
        } else {
            // PyTorch output is logits/probabilities, e.g., [0.1, 0.9]
            // We need to find the index of the highest value
            prediction = outputData.indexOf(Math.max(...outputData)); 
        }

        if (prediction === 1) {
            dom.resultBox.textContent = "CLAIM LIKELY (1)";
            dom.resultBox.className = 'claim';
        } else {
            dom.resultBox.textContent = "NO CLAIM LIKELY (0)";
            dom.resultBox.className = 'no-claim';
        }
    }
}

/**
 * Helper function to show/hide loading state
 */
function setLoading(dom, isLoading, message = "Loading...") {
    if (isLoading) {
        dom.resultBox.textContent = message;
        dom.resultBox.className = '';
        dom.predictButton.disabled = true;
        dom.loadingSpinner.style.display = 'block';
    } else {
        dom.predictButton.disabled = false;
        dom.loadingSpinner.style.display = 'none';
    }
}

// Wait for the DOM (and ort.min.js) to load before running main()
document.addEventListener("DOMContentLoaded", main);