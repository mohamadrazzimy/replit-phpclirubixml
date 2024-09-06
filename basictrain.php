<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$logger = new Screen();

$logger->info('Loading data into memory');

$training = Labeled::fromIterator(new CSV('dataset.csv'));

$testing = $training->randomize()->take(10);

$estimator = new KNearestNeighbors(5);

$logger->info('Training');

$estimator->train($training);

$logger->info('Making predictions');

$predictions = $estimator->predict($testing);
var_dump($predictions);

$metric = new Accuracy();
var_dump($metric);

$score = $metric->score($predictions, $testing->labels());
var_dump($score);

$logger->info("Accuracy is $score");