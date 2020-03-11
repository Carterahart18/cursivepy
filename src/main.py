import scripts.train_network as trainer

def main():
    trainer.train_network(iterations=10000, batch_size=500, epoch_length=120)

if __name__ == "__main__":
    main()
