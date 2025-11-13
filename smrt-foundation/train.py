from 


if __name__ == "__main__":
    # Settings
    BATCH_SIZE = 16
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FF = 256
    PREDICT_STEPS = 5
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    
    # 1. Create dummy dataset
    # Using "dummy" path to trigger internal dummy data generation
    # Include all kinetics columns
    dataset = SMRTSequenceDataset(
        parquet_path=data_path, 
        columns=['seq', 'fi', 'fp', 'ri', 'rp']
    )
    
    # 2. Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=cpc_collate_fn
    )

    # 3. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPCDNA(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        predict_steps=PREDICT_STEPS
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {device} with {len(dataset)} samples...")

    # 4. Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        
        for i, batch in tqdm(enumerate(dataloader)):
            seq_ids = batch['seq_ids'].to(device)
            kinetics = batch['kinetics'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            c, predictions = model(seq_ids, kinetics)
            
            # Calculate loss
            # We detach 'c' to use it as a target (z)
            # This prevents gradients from flowing back through the target side
            loss = cpc_loss(c.detach(), predictions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

    print("Training finished.")