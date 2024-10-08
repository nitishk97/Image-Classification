"foo" >> results.txt
rm results.txt

#Note: The values for 10%-100% of training data for Faces are as follows:
#45,90,135,180,225,270,315,360,405,451
#For digits:
#500,1000,1500,2000,2500,3000,3500,4000,4500,5000
$faceArray = 45
$digitArray = 500
$iterations = 1

#Choose which algorithms to run
$runNaiveBayes = $true
$runPerceptron = $true
$runkNearestNeighbor = $true
$start_time = Get-Date

#-----------------------------------------------------NAIVE BAYES-----------------------------------------------------------
if($runNaiveBayes)
{

    foreach ($amount in $faceArray) 
    {
        $start_faces = Get-Date
        For ($i=0; $i -lt $iterations; $i++)
        {
            "Running naiveBayes Faces: " + $amount
            python dataClassifier.py -c naiveBayes -d faces -t $amount -s 150 >> results.txt
        }
        $end_faces = Get-Date
        Write-host (New-TimeSpan -Start $start_faces -End $end_faces).TotalSeconds "seconds"
    }

    foreach ($amount in $digitArray) 
    {
        $start_digits = Get-Date
        For ($i=0; $i -lt $iterations; $i++)
        {
            "Running naiveBayes Digits: " + $amount
            python dataClassifier.py -c naiveBayes -d digits -t $amount -s 1000 -k .001 >> results.txt
        }
        $end_digits = Get-Date
        Write-host (New-TimeSpan -Start $start_digits -End $end_digits).TotalSeconds "seconds"
    }

}
#-----------------------------------------------------PERCEPTRON-----------------------------------------------------------
if($runPerceptron)
{

    foreach ($amount in $faceArray) 
    {
        $start_faces = Get-Date
        For ($i=0; $i -lt $iterations; $i++)
        {
            "Running Perceptron Faces: " + $amount
            python dataClassifier.py -c perceptron -d faces -t $amount -i 2 -s 150 >> results.txt
        }
        $end_faces = Get-Date
        Write-host (New-TimeSpan -Start $start_faces -End $end_faces).TotalSeconds "seconds"
    }

    foreach ($amount in $digitArray) 
    {
        $start_digits = Get-Date
        For ($i=0; $i -lt $iterations; $i++)
        {
            "Running Perceptron Digits: " + $amount
            python dataClassifier.py -c perceptron -d digits -t $amount -i 3 -s 1000 >> results.txt
        }
        $end_digits = Get-Date
        Write-host (New-TimeSpan -Start $start_digits -End $end_digits).TotalSeconds "seconds"
    }

}
#-------------------------------------------------k NEAREST NEIGHBOR--------------------------------------------------------
if($runkNearestNeighbor)
{

    foreach ($amount in $faceArray) 
    {
        $start_faces = Get-Date
        For ($i=0; $i -lt $iterations; $i++)
        {
            "Running k Nearest Neighbor Faces: " + $amount
            python dataClassifier.py -c kNN -d faces -t $amount -s 150 >> results.txt
        }
        $end_faces = Get-Date
        Write-host (New-TimeSpan -Start $start_faces -End $end_faces).TotalSeconds "seconds"
    }

    foreach ($amount in $digitArray) 
    {
        $start_digits = Get-Date
        For ($i=0; $i -lt $iterations; $i++)
        {
            "Running k Nearest Neighbor Digits: " + $amount
            python dataClassifier.py -c kNN -d digits -t $amount -s 1000 >> results.txt
        }
        $end_digits = Get-Date
        Write-host (New-TimeSpan -Start $start_digits -End $end_digits).TotalSeconds "seconds"
    }

}


$end_time = Get-Date
Write-host (New-TimeSpan -Start $start_time -End $end_time).TotalSeconds "seconds (TOTAL)"