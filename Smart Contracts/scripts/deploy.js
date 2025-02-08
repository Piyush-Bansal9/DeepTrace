async function main() {
  const [deployer] = await ethers.getSigners();

  console.log(
      "Deploying contracts with the account:",
      await deployer.getAddress()
  );


  const Contract = await ethers.getContractFactory("VideoStorage");
  const contract = await Contract.deploy();

  console.log("contract address on the sepolia testnet:", contract.target);

  await contract.waitForDeployment();
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
      console.error(error);
      process.exit(1);
  });