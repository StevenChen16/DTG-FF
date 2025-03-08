import torch
import torch.nn as nn
import torch.nn.functional as F

class DTGLayer(nn.Module):
    """Dynamic Temperature Goodness Layer
    
    Core ideas:
    1. Dynamically adjust temperature based on feature distribution
    2. Learn adaptive discrimination thresholds
    3. Use dynamic margin to distinguish positive and negative samples
    """
    def __init__(self, in_dim, out_dim, temp_min=0.1, temp_max=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        
        self.temp_min = temp_min
        self.temp_max = temp_max
        
    def calc_goodness(self, z):
        """Dynamic temperature goodness calculation
        
        Adaptively adjust temperature based on feature distribution
        """
        # Calculate feature statistics
        z_mean = z.mean(dim=0, keepdim=True)
        z_std = z.std(dim=0, keepdim=True)
        
        # Feature clarity (with numerical stability)
        feature_clarity = torch.clamp(z_std / (torch.abs(z_mean) + 1e-6), min=0.1, max=10.0).mean()
        temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(feature_clarity)
        
        # Calculate goodness score (using L2 norm)
        z_norm = F.normalize(z, p=2, dim=1)  # First normalize features
        z_scaled = z_norm / (temp + 1e-6)
        goodness = torch.sum(z_scaled ** 2, dim=1)
        
        return goodness, temp
    
    def forward(self, x):
        """Forward propagation"""
        # Feature transformation
        z = self.linear(x)
        z = self.bn(z)
        z = F.relu(z)  # Maintain non-negativity
        
        if self.training:
            # Training mode: Calculate goodness and related statistics
            goodness, temp = self.calc_goodness(z)
            threshold = F.softplus(self.threshold)  # Ensure threshold is positive
            return {
                "features": z,
                "goodness": goodness,
                "temperature": temp,
                "threshold": threshold
            }
        else:
            # Test mode: Only return features
            return z

class FF_DTG_Model(nn.Module):
    """Dynamic Temperature Goodness Forward-Forward Model"""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # Network structure
        self.layers = nn.ModuleList()
        for i in range(opt.model.num_layers):
            in_dim = opt.input_dim if i == 0 else opt.model.hidden_dim
            self.layers.append(DTGLayer(
                in_dim=in_dim,
                out_dim=opt.model.hidden_dim,
                temp_min=0.1,
                temp_max=2.0  # Lower maximum temperature
            ))
            
        # Classifier
        classifier_in_dim = opt.model.hidden_dim * (opt.model.num_layers - 1)
        self.classifier = nn.Linear(classifier_in_dim, opt.num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _calc_ff_loss(self, goodness, labels, temp, threshold):
        """Calculate FF loss
        
        Use dynamic margin and adaptive threshold
        """
        # Dynamic margin (with range limits)
        margin = torch.clamp(threshold * temp, min=0.1, max=2.0)
        
        # Process positive and negative samples separately
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        # Use log-sum-exp trick for numerical stability
        pos_diff = F.relu(threshold + margin - goodness[pos_mask])
        neg_diff = F.relu(goodness[neg_mask] - (threshold - margin))
        
        # Add weights to balance positive and negative samples
        pos_weight = 1.0 / (pos_mask.float().mean() + 1e-6)
        neg_weight = 1.0 / (neg_mask.float().mean() + 1e-6)
        
        pos_loss = pos_weight * pos_diff.mean()
        neg_loss = neg_weight * neg_diff.mean()
        
        return (pos_loss + neg_loss) / 2.0
    
    def forward(self, inputs, labels=None):
        """Forward propagation"""
        outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "FF_Loss": torch.zeros(1, device=self.opt.device),
            "Temperature": torch.zeros(1, device=self.opt.device)
        }
        
        if not isinstance(inputs, dict):
            # Non-training mode, return directly
            return outputs
            
        # Prepare data
        pos_z = inputs["pos_images"].reshape(inputs["pos_images"].shape[0], -1)
        neg_z = inputs["neg_images"].reshape(inputs["neg_images"].shape[0], -1)
        
        # Store intermediate features
        pos_features = []
        neg_features = []
        
        # Forward propagation - positive samples
        z = pos_z
        for i, layer in enumerate(self.layers):
            if self.training:
                layer_outputs = layer(z)
                z = layer_outputs["features"]
                pos_features.append(z)
                
                # Calculate FF loss
                pos_labels = torch.ones_like(layer_outputs["goodness"])
                ff_loss = self._calc_ff_loss(
                    layer_outputs["goodness"], 
                    pos_labels, 
                    layer_outputs["temperature"], 
                    layer_outputs["threshold"]
                )
                outputs["FF_Loss"] += ff_loss
                outputs["Temperature"] += layer_outputs["temperature"]
                
                # Calculate accuracy
                with torch.no_grad():
                    outputs[f"ff_accuracy_layer_{i}"] = (
                        layer_outputs["goodness"] > layer_outputs["threshold"]
                    ).float().mean()
            else:
                z = layer(z)
                pos_features.append(z)
        
        # Forward propagation - negative samples
        z = neg_z
        for i, layer in enumerate(self.layers):
            if self.training:
                layer_outputs = layer(z)
                z = layer_outputs["features"]
                neg_features.append(z)
                
                # Calculate FF loss
                neg_labels = torch.zeros_like(layer_outputs["goodness"])
                ff_loss = self._calc_ff_loss(
                    layer_outputs["goodness"], 
                    neg_labels, 
                    layer_outputs["temperature"], 
                    layer_outputs["threshold"]
                )
                outputs["FF_Loss"] += ff_loss
            else:
                z = layer(z)
                neg_features.append(z)
        
        # Classification task
        if "neutral_sample" in inputs and "class_labels" in labels:
            neutral_z = inputs["neutral_sample"].reshape(inputs["neutral_sample"].shape[0], -1)
            features = []
            
            # Extract features
            z = neutral_z
            for layer in self.layers[:-1]:  # Skip the last layer
                if self.training:
                    z = layer(z)["features"]
                else:
                    z = layer(z)
                features.append(z)
            
            # Classification prediction
            concat_features = torch.cat(features, dim=1)
            logits = self.classifier(concat_features)
            cls_loss = F.cross_entropy(logits, labels["class_labels"])
            outputs["Loss"] = outputs["FF_Loss"] + cls_loss
            
            # Calculate classification accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                outputs["classification_accuracy"] = (preds == labels["class_labels"]).float().mean()
        else:
            outputs["Loss"] = outputs["FF_Loss"]
        
        return outputs

class FF_DTG_Config:
    """Dynamic Temperature Goodness Configuration"""
    def __init__(self):
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Input settings
        self.input = type('', (), {})()
        self.input.path = "data"
        self.input.batch_size = 128
        self.input.dataset = "mnist"  # 'mnist' or 'cifar10'
        
        # Model settings
        self.model = type('', (), {})()
        self.model.hidden_dim = 2048
        self.model.num_layers = 4
        
        # Training settings
        self.training = type('', (), {})()
        self.training.epochs = 100
        self.training.learning_rate = 1e-3
        self.training.weight_decay = 1e-4
        self.training.momentum = 0.9
        
        # Set input dimensions based on dataset
        self.input_dim = 784  # Default for MNIST
        self.num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes