import scripts.train_network as trainer
import scripts.test_network as tester


def main():
    # trainer.train_network(iterations=10000, batch_size=500, epoch_length=120)
    # trainer.train_network(iterations=120, batch_size=500, epoch_length=120)
    tester.test_network(iterations=100, batch_size=500)

if __name__ == "__main__":
    main()
