import numpy as np
def generate_fruit_dataset(num_samples):
    # Define features and classes
    colors = ['red', 'yellow', 'green', 'orange', 'purple']  # Adding more color types
    sizes = ['small', 'medium', 'large']
    classes = ['apple', 'banana', 'orange', 'grape']

    # Define realistic data based on fruit characteristics
    inputs = []
    targets = []
    for _ in range(num_samples):
        color = np.random.choice(colors)
        size = np.random.choice(sizes)

        # Assign shape based on fruit type
        if color == 'yellow':
            shape = 'curved'  # Bananas are curved
            fruit_class = 'banana'
        elif color == 'red':
            shape = 'oval'  # Apples are oval
            fruit_class = 'apple'
        elif color == 'orange':
            shape = 'round'  # Oranges are round
            fruit_class = 'orange'
        elif color == 'green' or color == 'purple':
            shape = 'round'  # Grapes are round
            fruit_class = 'grape'
        else:
            # For other colors, assign shapes randomly
            if np.random.rand() < 0.8:  # Most fruits are round or oval
                shape = np.random.choice(['round', 'oval'])
            else:
                shape = 'curved' if np.random.rand() < 0.5 else np.random.choice(['round', 'oval'])  # Most bananas are curved

            # Assign class randomly for other colors
            fruit_class = np.random.choice(['apple', 'banana', 'orange', 'grape'])

        inputs.append([color, size, shape])
        targets.append(fruit_class)

    return inputs, targets

# Generate multi-class fruit dataset with 1000 samples
num_samples = 1000
inputs, targets = generate_fruit_dataset(num_samples)

print("Multi-class fruit dataset generated successfully!")
print("Example input data:", inputs[:5])
print("Example targets:", targets[:5])
# Encoding categorical features
color_encoder = LabelEncoder()
size_encoder = LabelEncoder()
shape_encoder = LabelEncoder()

# Extract features from inputs
colors = [item[0] for item in inputs]
sizes = [item[1] for item in inputs]
shapes = [item[2] for item in inputs]

# Fit encoders
color_encoder.fit(colors)
size_encoder.fit(sizes)
shape_encoder.fit(shapes)

# Transform categorical features
colors_encoded = color_encoder.transform(colors)
sizes_encoded = size_encoder.transform(sizes)
shapes_encoded = shape_encoder.transform(shapes)

inputs_encoded = np.column_stack((colors_encoded, sizes_encoded, shapes_encoded))

# Create and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(inputs_encoded, targets)

# Make predictions for probabilities
proba_predictions = knn_model.predict_proba(inputs_encoded)

# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(targets, proba_predictions[:, 1], pos_label='banana')
roc_auc = auc(fpr, tpr)

# Plot micro-average ROC curve
fig = px.area(
    x=fpr,
    y=tpr,
    title="ROC Curve (Micro-average)"
)

fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='green', dash='dash'))

fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    margin=dict(l=10, r=10, t=25, b=10),
)

fig.show()

# Make a prediction function
def guess_fruit(knn_model, color_encoder, size_encoder, shape_encoder):
    # Get user input for fruit characteristics
    color = input("Enter the color of the fruit (red, yellow, green, orange, purple): ").lower()
    size = input("Enter the size of the fruit (small, medium, large): ").lower()
    shape = input("Enter the shape of the fruit (round, oval, curved): ").lower()

    # Transform user input
    color_encoded = color_encoder.transform([color])[0]
    size_encoded = size_encoder.transform([size])[0]
    shape_encoded = shape_encoder.transform([shape])[0]

    # Make a prediction
    prediction = knn_model.predict([[color_encoded, size_encoded, shape_encoded]])
    print("Predicted fruit class:", prediction[0])

# Example usage
guess_fruit(knn_model, color_encoder, size_encoder, shape_encoder)
